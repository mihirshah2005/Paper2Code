"""dataset_loader.py

This module defines the DatasetLoader class which loads and preprocesses
datasets for translation and parsing tasks. It leverages SentencePiece for
subword tokenization, performs fixed random splits into train/val/test,
and constructs PyTorch DataLoader objects with a custom dynamic batching
sampler based on token count. All configuration values are read from a
configuration dictionary parsed from config.yaml.

Author: [Your Name]
Date: [Current Date]
"""

import os
import random
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler, Subset
import sentencepiece as spm

# Setup logging.
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)


###############################################################################
# SentencePiece Helper Functions
###############################################################################
def train_sentencepiece(texts, vocab_size: int, model_prefix: str):
    """
    Train a SentencePiece model using the provided texts.
    The model is saved as {model_prefix}_vocab{vocab_size}.model.
    A temporary input file is created and then removed.
    """
    model_file = f"{model_prefix}_vocab{vocab_size}.model"
    input_file = f"{model_prefix}_input.txt"
    with open(input_file, "w", encoding="utf8") as f:
        for line in texts:
            f.write(line + "\n")
    spm.SentencePieceTrainer.Train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=1.0,
        model_type='unigram'
    )
    logger.info(f"Trained SentencePiece model and saved to {model_file}")
    os.remove(input_file)


def get_sentencepiece_processor(texts, vocab_size: int, model_prefix: str) -> spm.SentencePieceProcessor:
    """
    Get a SentencePieceProcessor. If a model file exists, load it;
    otherwise, train a new SentencePiece model.
    """
    model_file = f"{model_prefix}_vocab{vocab_size}.model"
    processor = spm.SentencePieceProcessor()
    if os.path.isfile(model_file):
        try:
            processor.Load(model_file)
            logger.info(f"Loaded SentencePiece model from {model_file}")
        except Exception as e:
            logger.error(f"Error loading {model_file}: {e}. Retraining model.")
            train_sentencepiece(texts, vocab_size, model_prefix)
            processor.Load(model_file)
    else:
        train_sentencepiece(texts, vocab_size, model_prefix)
        processor.Load(model_file)
    return processor


###############################################################################
# Data Collation and Batching
###############################################################################
def custom_collate_fn(batch, pad_id: int, task="translation"):
    """
    Collate a list of examples into a padded batch.
    For translation tasks, each example is a dict with keys "src" and "tgt".
    For parsing tasks, each example is expected to have key "input".
    Returns a dict containing padded sequences and corresponding masks.
    """
    if task == "translation":
        src_sequences = [torch.tensor(item["src"], dtype=torch.long) for item in batch]
        tgt_sequences = [torch.tensor(item["tgt"], dtype=torch.long) for item in batch]
        src_padded = torch.nn.utils.rnn.pad_sequence(src_sequences, batch_first=True, padding_value=pad_id)
        tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_sequences, batch_first=True, padding_value=pad_id)
        src_mask = (src_padded != pad_id).long()
        tgt_mask = (tgt_padded != pad_id).long()
        return {"src": src_padded, "src_mask": src_mask, "tgt": tgt_padded, "tgt_mask": tgt_mask}
    elif task == "parsing":
        input_sequences = [torch.tensor(item["input"], dtype=torch.long) for item in batch]
        input_padded = torch.nn.utils.rnn.pad_sequence(input_sequences, batch_first=True, padding_value=pad_id)
        input_mask = (input_padded != pad_id).long()
        return {"input": input_padded, "input_mask": input_mask}
    else:
        raise ValueError("Unknown task in collate function.")


class DynamicBatchSampler(Sampler):
    """
    A custom Sampler that groups examples from the dataset into batches such that
    the cumulative token count (for src and tgt sequences separately in translation,
    or for input in parsing) does not exceed a specified threshold.
    """
    def __init__(self, data_source, batch_size_tokens: int, task="translation", shuffle: bool = True):
        self.data_source = data_source
        self.batch_size_tokens = batch_size_tokens
        self.task = task
        self.shuffle = shuffle
        self.indices = list(range(len(data_source)))
        if self.shuffle:
            random.seed(42)
            random.shuffle(self.indices)

    def __iter__(self):
        batch = []
        total_tokens_src = 0
        total_tokens_tgt = 0
        total_tokens_input = 0
        for idx in self.indices:
            item = self.data_source[idx]
            if self.task == "translation":
                src_len = len(item["src"])
                tgt_len = len(item["tgt"])
                if not batch:
                    batch.append(idx)
                    total_tokens_src = src_len
                    total_tokens_tgt = tgt_len
                else:
                    if (total_tokens_src + src_len <= self.batch_size_tokens) and (total_tokens_tgt + tgt_len <= self.batch_size_tokens):
                        batch.append(idx)
                        total_tokens_src += src_len
                        total_tokens_tgt += tgt_len
                    else:
                        yield batch
                        batch = [idx]
                        total_tokens_src = src_len
                        total_tokens_tgt = tgt_len
            else:  # parsing task: use "input"
                item_len = len(item["input"])
                if not batch:
                    batch.append(idx)
                    total_tokens_input = item_len
                else:
                    if total_tokens_input + item_len <= self.batch_size_tokens:
                        batch.append(idx)
                        total_tokens_input += item_len
                    else:
                        yield batch
                        batch = [idx]
                        total_tokens_input = item_len
        if batch:
            yield batch

    def __len__(self):
        # This is an approximation.
        return len(self.data_source) // 1


###############################################################################
# Tree Linearization for Parsing Tasks
###############################################################################
def linearize_tree(tree_str: str) -> str:
    """
    Linearize a bracketed tree structure using a simple tokenization algorithm.
    This function tokenizes parentheses and labels, then returns a space-separated string.
    """
    tokens = []
    token = ''
    for char in tree_str:
        if char in ('(', ')'):
            if token:
                tokens.append(token)
                token = ''
            tokens.append(char)
        elif char.isspace():
            if token:
                tokens.append(token)
                token = ''
        else:
            token += char
    if token:
        tokens.append(token)
    return " ".join(tokens)


###############################################################################
# Dummy Dataset Loading Functions
###############################################################################
def load_translation_dataset(dataset_name: str) -> list:
    """
    Simulate loading a translation dataset.
    Returns a list of dictionaries with keys "src_text" and "tgt_text".
    In practice, this function should load the actual WMT14 dataset.
    """
    examples = []
    for i in range(100):
        src_text = f"This is source sentence number {i}."
        tgt_text = f"This is target sentence number {i}."
        examples.append({"src_text": src_text, "tgt_text": tgt_text})
    return examples


def load_parsing_dataset(dataset_name: str) -> list:
    """
    Simulate loading a parsing dataset.
    Returns a list of dictionaries with key "tree" containing bracketed tree strings.
    In practice, this function should load the WSJ Penn Treebank.
    """
    examples = []
    for i in range(50):
        tree_str = f"(S (NP This) (VP is (NP a sentence {i})))"
        examples.append({"tree": tree_str})
    return examples


###############################################################################
# Dataset Classes for Processed Examples
###############################################################################
class TranslationDataset(Dataset):
    """
    A Dataset class for translation tasks.
    Given raw examples with "src_text" and "tgt_text", this class tokenizes the text
    using a shared SentencePiece processor and produces examples with token IDs.
    """
    def __init__(self, raw_examples: list, sp_processor: spm.SentencePieceProcessor):
        self.examples = []
        self.sp_processor = sp_processor
        for ex in raw_examples:
            src_tokens = sp_processor.EncodeAsIds(ex["src_text"])
            tgt_tokens = sp_processor.EncodeAsIds(ex["tgt_text"])
            self.examples.append({"src": src_tokens, "tgt": tgt_tokens})

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class ParsingDatasetProcessed(Dataset):
    """
    A Dataset class for parsing tasks.
    Given raw examples with a bracketed tree string under key "tree",
    this class linearizes the tree and tokenizes it using a SentencePiece processor.
    """
    def __init__(self, raw_examples: list, sp_processor: spm.SentencePieceProcessor):
        self.examples = []
        self.sp_processor = sp_processor
        for ex in raw_examples:
            linear_tree = linearize_tree(ex["tree"])
            tokens = sp_processor.EncodeAsIds(linear_tree)
            self.examples.append({"input": tokens})

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


###############################################################################
# Main DatasetLoader Class
###############################################################################
class DatasetLoader:
    """
    The DatasetLoader class loads datasets according to the provided configuration.
    It supports both translation and parsing tasks.
    Upon calling load_data(), it performs the following:
      - Loads raw text examples.
      - Trains/loads a SentencePiece model for tokenization.
      - Tokenizes the raw examples.
      - Splits the tokenized examples into train/validation/test sets (80/10/10 split).
      - Constructs PyTorch DataLoader objects with a custom dynamic batching sampler.
    """
    def __init__(self, config: dict, task: str = "translation"):
        self.config = config
        self.task = task  # Expected values: "translation" or "parsing"
        self.random_seed: int = 42  # Fixed seed for reproducibility
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)

        self.batch_token_limit: int = config["training"].get("batch_size_tokens", 25000)

        if self.task == "translation":
            self.dataset_info = self.config["data"]["translation"]
            # Default to English-German task if available.
            self.dataset_name: str = self.dataset_info.get("english_german", "WMT14_en_de")
            self.vocab_size: int = self.dataset_info.get("vocabulary", {}).get("english_german", 37000)
            # Use shared vocabulary approach (default)
            self.shared_vocab: bool = True
        elif self.task == "parsing":
            self.dataset_info = self.config["data"]["parsing"]
            self.dataset_name: str = self.dataset_info.get("dataset", "WSJ_Penn_Treebank")
            # Use wsj_only vocabulary as default
            self.vocab_size: int = self.dataset_info.get("vocabulary", {}).get("wsj_only", 16000)
        else:
            raise ValueError("Unsupported task type. Choose 'translation' or 'parsing'.")

        # Define directory to store SentencePiece models.
        self.spm_model_dir: str = "models"
        os.makedirs(self.spm_model_dir, exist_ok=True)

    def load_data(self) -> tuple:
        """
        Load the dataset, tokenize the text, split into train/validation/test sets,
        and create PyTorch DataLoader objects with dynamic batching.
        Returns:
            (train_loader, val_loader, test_loader)
        """
        if self.task == "translation":
            raw_examples = load_translation_dataset(self.dataset_name)
            # Build a combined corpus of source and target texts.
            corpus_texts = [ex["src_text"] for ex in raw_examples] + [ex["tgt_text"] for ex in raw_examples]
            model_prefix = os.path.join(self.spm_model_dir, f"spm_{self.dataset_name}")
            sp_processor = get_sentencepiece_processor(corpus_texts, self.vocab_size, model_prefix)
            pad_id: int = sp_processor.pad_id() if sp_processor.pad_id() is not None else 0
            dataset = TranslationDataset(raw_examples, sp_processor)
        else:  # parsing task
            raw_examples = load_parsing_dataset(self.dataset_name)
            corpus_texts = [linearize_tree(ex["tree"]) for ex in raw_examples]
            model_prefix = os.path.join(self.spm_model_dir, f"spm_{self.dataset_name}")
            sp_processor = get_sentencepiece_processor(corpus_texts, self.vocab_size, model_prefix)
            pad_id: int = sp_processor.pad_id() if sp_processor.pad_id() is not None else 0
            dataset = ParsingDatasetProcessed(raw_examples, sp_processor)

        # Create train/val/test splits using an 80/10/10 split.
        total_examples = len(dataset)
        indices = list(range(total_examples))
        random.shuffle(indices)
        train_split = int(0.8 * total_examples)
        val_split = int(0.9 * total_examples)
        train_indices = indices[:train_split]
        val_indices = indices[train_split:val_split]
        test_indices = indices[val_split:]
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        test_dataset = Subset(dataset, test_indices)

        # Create DataLoader objects with a custom dynamic batch sampler.
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=DynamicBatchSampler(train_dataset, self.batch_token_limit, task=self.task, shuffle=True),
            collate_fn=lambda batch: custom_collate_fn(batch, pad_id, task=self.task),
            num_workers=0
        )
        val_loader = DataLoader(
            val_dataset,
            batch_sampler=DynamicBatchSampler(val_dataset, self.batch_token_limit, task=self.task, shuffle=False),
            collate_fn=lambda batch: custom_collate_fn(batch, pad_id, task=self.task),
            num_workers=0
        )
        test_loader = DataLoader(
            test_dataset,
            batch_sampler=DynamicBatchSampler(test_dataset, self.batch_token_limit, task=self.task, shuffle=False),
            collate_fn=lambda batch: custom_collate_fn(batch, pad_id, task=self.task),
            num_workers=0
        )

        logger.info(f"Loaded {len(train_dataset)} training, {len(val_dataset)} validation, and {len(test_dataset)} test examples for task '{self.task}'.")
        return train_loader, val_loader, test_loader
