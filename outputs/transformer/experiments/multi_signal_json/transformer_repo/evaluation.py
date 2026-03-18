"""evaluation.py

This module implements the Evaluation class for running beam search–based inference 
on a Transformer model and computing evaluation metrics (BLEU for translation or F1 for parsing).
The inference loop is wrapped in a torch.no_grad() context to avoid gradient computation.
Beam search is used to generate predictions for each test example.

Author: [Your Name]
Date: [Current Date]
"""

import math
import time
import torch
import torch.nn.functional as F
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple
import logging

# Import the TransformerModel from the model module.
from model import TransformerModel

# Special token IDs; set default values.
SOS_TOKEN: int = 1
EOS_TOKEN: int = 2

# Configure module-level logging.
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)


class Evaluation:
    """
    Evaluation class that implements beam search decoding and computes evaluation metrics.
    It supports both translation and parsing tasks. For translation, the corpus BLEU score is
    computed; for parsing, the F1 score (based on tree spans) is computed.
    """
    def __init__(self, model: TransformerModel, test_loader: torch.utils.data.DataLoader, config: Dict[str, Any]) -> None:
        """
        Initializes the Evaluation instance.
        
        Args:
            model (TransformerModel): The trained Transformer model.
            test_loader (DataLoader): DataLoader for the test dataset.
            config (dict): Configuration dictionary (from config.yaml).
        """
        self.model: TransformerModel = model
        self.test_loader: torch.utils.data.DataLoader = test_loader
        self.config: Dict[str, Any] = config
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Determine the task based on keys in a sample batch.
        sample_batch = next(iter(test_loader))
        if "src" in sample_batch:
            self.task: str = "translation"
            self.inference_cfg: Dict[str, Any] = self.config.get("inference", {}).get("translation", {})
        elif "input" in sample_batch:
            self.task = "parsing"
            self.inference_cfg = self.config.get("inference", {}).get("parsing", {})
        else:
            raise ValueError("Test loader batch does not contain expected keys for evaluation.")

        # Get inference configuration parameters with default values.
        self.beam_size: int = self.inference_cfg.get("beam_size", 4)
        self.length_penalty: float = self.inference_cfg.get("length_penalty", 0.6)
        self.max_length_offset: int = self.inference_cfg.get("max_length_offset", 50)

        # Attempt to retrieve the SentencePiece processor from the test dataset (if available).
        self.sp_processor = getattr(self.test_loader.dataset, "sp_processor", None)

    def _decode_tokens(self, tokens: List[int]) -> str:
        """
        Converts a list of token IDs into a decoded text string.
        It removes the SOS token if present and truncates the sequence at EOS.
        
        Args:
            tokens (List[int]): List of token IDs.
            
        Returns:
            str: Decoded string.
        """
        if tokens and tokens[0] == SOS_TOKEN:
            tokens = tokens[1:]
        if EOS_TOKEN in tokens:
            tokens = tokens[:tokens.index(EOS_TOKEN)]
        if self.sp_processor is not None:
            try:
                decoded_text = self.sp_processor.DecodeIds(tokens)
            except Exception as e:
                logger.error(f"Error decoding tokens with SentencePiece: {e}")
                decoded_text = " ".join(map(str, tokens))
        else:
            decoded_text = " ".join(map(str, tokens))
        return decoded_text

    def _beam_search(self, src: torch.Tensor, src_mask: torch.Tensor) -> List[int]:
        """
        Performs beam search to generate a prediction for a single source example.
        
        Args:
            src (torch.Tensor): Source tensor of shape (1, src_seq_len).
            src_mask (torch.Tensor): Source mask tensor of shape (1, src_seq_len).
            
        Returns:
            List[int]: The predicted token ID sequence.
        """
        # Maximum output length = source length + max_length_offset.
        max_len: int = src.size(1) + self.max_length_offset

        # Precompute the encoder memory from the source.
        self.model.eval()
        with torch.no_grad():
            src_emb = self.model.embed(src) * self.model.scale  # (1, src_seq_len, d_model)
            src_emb = self.model.pos_encoding(src_emb)
            encoder_mask = src_mask.unsqueeze(1).unsqueeze(2) if src_mask is not None else None
            memory = self.model.encoder(src_emb, mask=encoder_mask)

        # Initialization: one hypothesis starting with SOS_TOKEN.
        active_hyps: List[Dict[str, Any]] = [{"sequence": [SOS_TOKEN], "log_prob": 0.0}]
        finished_hyps: List[Dict[str, Any]] = []

        for t in range(1, max_len + 1):
            all_candidates: List[Dict[str, Any]] = []
            if not active_hyps:
                break  # No active hypotheses left.
            current_beam_size: int = len(active_hyps)
            # Prepare the current hypotheses as target sequences.
            sequences: List[List[int]] = [hypo["sequence"] for hypo in active_hyps]
            tgt_batch: torch.Tensor = torch.tensor(sequences, dtype=torch.long, device=self.device)  # (B, t)
            # Replicate the source and its mask for the beam.
            repeated_src: torch.Tensor = src.repeat(current_beam_size, 1)
            repeated_src_mask: Optional[torch.Tensor] = src_mask.repeat(current_beam_size, 1) if src_mask is not None else None
            
            # Run the model forward: obtain logits for the target batch.
            with torch.no_grad():
                logits = self.model(repeated_src, tgt_batch, src_mask=repeated_src_mask, tgt_mask=None)
            # Extract logits for the last time step.
            last_logits: torch.Tensor = logits[:, -1, :]  # (B, vocab_size)
            log_probs: torch.Tensor = F.log_softmax(last_logits, dim=-1)  # (B, vocab_size)
            
            # Expand each hypothesis by exploring top candidates.
            for i, hypo in enumerate(active_hyps):
                topk_log_probs, topk_indices = log_probs[i].topk(self.beam_size)
                for j in range(self.beam_size):
                    new_token: int = topk_indices[j].item()
                    new_log_prob: float = hypo["log_prob"] + topk_log_probs[j].item()
                    new_sequence: List[int] = hypo["sequence"] + [new_token]
                    # Apply length penalty: LP = ((5 + length)/6)^alpha.
                    lp: float = ((5 + len(new_sequence)) / 6) ** self.length_penalty
                    norm_score: float = new_log_prob / lp
                    candidate: Dict[str, Any] = {
                        "sequence": new_sequence,
                        "log_prob": new_log_prob,
                        "norm_score": norm_score
                    }
                    all_candidates.append(candidate)
            
            # Sort candidates by normalized score in descending order.
            all_candidates.sort(key=lambda x: x["norm_score"], reverse=True)
            new_active_hyps: List[Dict[str, Any]] = []
            for candidate in all_candidates:
                if candidate["sequence"][-1] == EOS_TOKEN:
                    finished_hyps.append(candidate)
                else:
                    new_active_hyps.append(candidate)
                if len(new_active_hyps) >= self.beam_size:
                    break
            active_hyps = new_active_hyps
            if not active_hyps:
                break  # Exit if there are no active hypotheses.

        # Final selection: choose the best hypothesis among finished ones (or active if none finished).
        if finished_hyps:
            best_hyp: Dict[str, Any] = max(finished_hyps, key=lambda x: x["norm_score"])
        elif active_hyps:
            best_hyp = max(active_hyps, key=lambda x: x["norm_score"])
        else:
            best_hyp = {"sequence": []}
        return best_hyp["sequence"]

    def evaluate(self) -> Dict[str, float]:
        """
        Runs evaluation on the test dataset using beam search decoding.
        For translation tasks, computes the corpus BLEU score.
        For parsing tasks, computes the F1 score based on the extracted tree spans.
        
        Returns:
            dict: A dictionary containing the computed metric, e.g., {"BLEU": value} or {"F1": value}.
        """
        self.model.eval()
        predictions: List[str] = []
        references: List[str] = []

        with torch.no_grad():
            for batch in self.test_loader:
                # Determine batch size based on task.
                if "src" in batch:
                    batch_size: int = batch["src"].size(0)
                elif "input" in batch:
                    batch_size = batch["input"].size(0)
                else:
                    raise ValueError("Batch does not contain expected keys for evaluation.")
                # Process each example in the batch individually.
                for i in range(batch_size):
                    if self.task == "translation":
                        src = batch["src"][i].unsqueeze(0).to(self.device)  # (1, src_seq_len)
                        src_mask = batch["src_mask"][i].unsqueeze(0).to(self.device)  # (1, src_seq_len)
                        tgt = batch["tgt"][i]  # Gold target token IDs.
                    else:
                        src = batch["input"][i].unsqueeze(0).to(self.device)
                        src_mask = batch["input_mask"][i].unsqueeze(0).to(self.device)
                        tgt = batch["input"][i]  # Gold tree token IDs.
                    
                    # Generate prediction via beam search.
                    pred_ids: List[int] = self._beam_search(src, src_mask)
                    pred_text: str = self._decode_tokens(pred_ids)
                    predictions.append(pred_text)
                    
                    # Prepare reference text by removing padding tokens (assumed pad token = 0).
                    ref_ids: List[int] = tgt.tolist()
                    ref_ids = [token for token in ref_ids if token != 0]
                    ref_text: str = self._decode_tokens(ref_ids)
                    references.append(ref_text)
        
        # Compute evaluation metrics.
        if self.task == "translation":
            bleu: float = self._compute_bleu(references, predictions)
            logger.info(f"Evaluation BLEU score: {bleu:.2f}")
            return {"BLEU": bleu}
        else:
            f1: float = self._compute_f1(references, predictions)
            logger.info(f"Evaluation F1 score: {f1:.2f}")
            return {"F1": f1}

    def _compute_bleu(self, references: List[str], candidates: List[str]) -> float:
        """
        Computes the corpus BLEU score using modified n-gram precision and brevity penalty.
        
        Args:
            references (List[str]): List of reference sentences.
            candidates (List[str]): List of generated candidate sentences.
        
        Returns:
            float: The computed BLEU score (percentage).
        """
        max_n: int = 4
        total_candidate_length: int = 0
        total_reference_length: int = 0
        precisions: List[float] = [0.0 for _ in range(max_n)]
        counts: List[int] = [0 for _ in range(max_n)]
        
        for ref_sentence, cand_sentence in zip(references, candidates):
            ref_tokens = ref_sentence.split()
            cand_tokens = cand_sentence.split()
            total_candidate_length += len(cand_tokens)
            total_reference_length += len(ref_tokens)
            for n in range(1, max_n + 1):
                ref_ngrams = Counter([tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens) - n + 1)])
                cand_ngrams = Counter([tuple(cand_tokens[i:i+n]) for i in range(len(cand_tokens) - n + 1)])
                overlap = sum(min(count, ref_ngrams.get(ngram, 0)) for ngram, count in cand_ngrams.items())
                precisions[n-1] += overlap
                counts[n-1] += max(len(cand_tokens) - n + 1, 0)
        
        p_ns: List[float] = []
        for i in range(max_n):
            p = (precisions[i] / counts[i]) if counts[i] > 0 else 0.0
            p_ns.append(p)
        
        # If any n-gram precision is zero, set geometric mean to zero.
        if min(p_ns) == 0:
            geo_mean = 0.0
        else:
            geo_mean = math.exp(sum(math.log(p) for p in p_ns) / max_n)
        
        # Brevity penalty.
        bp: float = 1.0
        if total_candidate_length < total_reference_length and total_candidate_length > 0:
            bp = math.exp(1 - total_reference_length / total_candidate_length)
        
        bleu = bp * geo_mean * 100  # Expressed as percentage.
        return bleu

    def _extract_spans(self, tree_str: str) -> set:
        """
        Extracts constituent spans from a linearized bracketed tree string.
        
        Args:
            tree_str (str): The linearized tree string.
        
        Returns:
            set: A set of (start, end) tuples representing constituent spans.
        """
        tokens = tree_str.split()
        spans = set()
        stack = []
        index = 0
        for token in tokens:
            if token == "(":
                # Mark the beginning of a constituent.
                stack.append(index)
            elif token == ")":
                if stack:
                    start = stack.pop()
                    end = index
                    if end - start > 1:
                        spans.add((start, end))
            else:
                index += 1
        return spans

    def _compute_f1(self, references: List[str], candidates: List[str]) -> float:
        """
        Computes the F1 score for parsing by comparing predicted and reference tree spans.
        
        Args:
            references (List[str]): List of gold-standard tree strings.
            candidates (List[str]): List of predicted tree strings.
        
        Returns:
            float: F1 score (percentage).
        """
        total_predicted = 0
        total_gold = 0
        total_overlap = 0
        for ref, cand in zip(references, candidates):
            gold_spans = self._extract_spans(ref)
            pred_spans = self._extract_spans(cand)
            total_gold += len(gold_spans)
            total_predicted += len(pred_spans)
            total_overlap += len(gold_spans.intersection(pred_spans))
        precision = total_overlap / total_predicted if total_predicted > 0 else 0.0
        recall = total_overlap / total_gold if total_gold > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        return f1 * 100  # Expressed as percentage.


if __name__ == "__main__":
    # Example usage: Evaluate the Transformer model.
    import argparse
    import yaml
    from dataset_loader import DatasetLoader
    from trainer import Trainer  # Not used directly here, but part of the overall system.
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(description="Evaluate Transformer model using beam search decoding.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file.")
    parser.add_argument("--task", type=str, default="translation", choices=["translation", "parsing"], help="Evaluation task type.")
    args = parser.parse_args()

    # Load configuration.
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Load test data using the DatasetLoader.
    dataset_loader = DatasetLoader(config, task=args.task)
    _, _, test_loader = dataset_loader.load_data()

    # Determine vocabulary size based on task.
    if args.task == "translation":
        vocab_size = config["data"]["translation"]["vocabulary"].get("english_german", 37000)
    else:
        vocab_size = config["data"]["parsing"]["vocabulary"].get("wsj_only", 16000)

    # Initialize the Transformer model (using base model configuration by default).
    model = TransformerModel(config, vocab_size=vocab_size, task=args.task, model_type="base")

    # Create an Evaluation instance.
    evaluator = Evaluation(model, test_loader, config)
    metrics = evaluator.evaluate()
    print("Evaluation metrics:", metrics)
