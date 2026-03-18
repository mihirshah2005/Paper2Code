"""evaluation.py

This module implements the Evaluation class, which is responsible for evaluating a trained
model using domain-specific metrics. It supports evaluation for Computer Vision (CV),
Natural Language Processing (NLP), and Graph domains. The Evaluation class uses the configuration
from "config.yaml" to determine the domain, select the proper metric set, and handle domain‑specific
output processing during evaluation.

Key Features:
    - Domain determination: Uses config["domain"] if specified; otherwise raises an error to
      avoid ambiguity.
    - Forward pass in evaluation mode with torch.no_grad(), accumulation of predictions and true labels.
    - Domain-specific output processing:
         • For CV/NLP: Applies argmax to logits to get predicted class labels.
         • For Graph: Applies sigmoid (or softmax for 2-class outputs) based on the configuration flag
           "graph_apply_sigmoid" (default True) to yield continuous probability estimates.
    - Metric computation using scikit‑learn (accuracy_score, f1_score, matthews_corrcoef, roc_auc_score)
      and scipy.stats (spearmanr) for Spearman correlation.
    - Optional recording of evaluation time and GPU memory (if enabled in config["evaluation"]["record_complexity"]).

Usage:
    evaluation = Evaluation(model, data, config)
    results = evaluation.evaluate()
"""

import time
import logging
from typing import Any, Dict, List

import torch
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from scipy.stats import spearmanr

# Set up module-level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Evaluation:
    """
    Evaluation class to perform inference on the evaluation dataset and compute performance metrics.

    Args:
        model (torch.nn.Module): The trained model; must have a forward() method.
        data (Dict[str, Any]): A dictionary containing DataLoader objects for evaluation.
                                Expected keys are "test" or alternatively "val".
        config (Dict[str, Any]): Experiment configuration dictionary loaded from config.yaml.
                                 It must include an "evaluation" section and training settings.
                                 For unambiguous domain determination, an explicit "domain" key is required.
    """

    def __init__(self, model: torch.nn.Module, data: Dict[str, Any], config: Dict[str, Any]) -> None:
        self.model = model
        self.data = data
        self.config = config

        # Determine domain:
        # 1. If config explicitly defines "domain", use that.
        if "domain" in self.config and isinstance(self.config["domain"], str):
            self.domain: str = self.config["domain"].strip().lower()
            logger.info(f"Using explicit domain from config: {self.domain}")
        else:
            # 2. Check training configuration blocks
            training_config = self.config.get("training", {})
            active_domains = [k for k in training_config.keys() if training_config.get(k)]
            if len(active_domains) == 1:
                self.domain = active_domains[0].strip().lower()
                logger.info(f"Implicitly determined domain from training config: {self.domain}")
            else:
                error_msg = (
                    "Ambiguous training domains detected. Please explicitly set the 'domain' field in "
                    "the configuration to one of 'cv', 'nlp', or 'graph'."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

        # Retrieve metrics for the domain from evaluation config.
        eval_metrics = self.config.get("evaluation", {}).get("metrics", {})
        domain_metrics = eval_metrics.get(self.domain)
        if domain_metrics is None:
            error_msg = f"No evaluation metrics specified for domain '{self.domain}' in configuration."
            logger.error(error_msg)
            raise ValueError(error_msg)
        # Ensure metrics is a list.
        if isinstance(domain_metrics, str):
            self.metrics: List[str] = [domain_metrics]
        elif isinstance(domain_metrics, list):
            self.metrics = domain_metrics
        else:
            error_msg = "Evaluation metrics configuration must be a string or a list of strings."
            logger.error(error_msg)
            raise ValueError(error_msg)
        logger.info(f"Evaluation metrics for domain '{self.domain}': {self.metrics}")

        # For graph domain, get flag for applying sigmoid transformation.
        self.graph_apply_sigmoid: bool = True
        if self.domain == "graph":
            self.graph_apply_sigmoid = self.config.get("evaluation", {}).get("graph_apply_sigmoid", True)
            logger.info(f"Graph evaluation: graph_apply_sigmoid set to {self.graph_apply_sigmoid}")

        # Record complexity flags.
        record_complexity = self.config.get("evaluation", {}).get("record_complexity", {})
        self.record_eval_time: bool = bool(record_complexity.get("training_time", False))
        self.record_gpu_memory: bool = bool(record_complexity.get("gpu_memory", False))

        # Device: use model's device (assume at least one parameter exists)
        self.device = next(self.model.parameters()).device

    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluates the model on the evaluation dataset and computes domain-specific metrics.

        Returns:
            Dict[str, Any]: A dictionary containing computed performance metrics and, if enabled,
                            evaluation time and GPU memory consumption.
        """
        self.model.eval()
        eval_start_time = time.time()

        # Choose evaluation dataset: prefer "test", fallback to "val"
        if "test" in self.data:
            eval_loader = self.data["test"]
            logger.info("Using 'test' dataset for evaluation.")
        elif "val" in self.data:
            eval_loader = self.data["val"]
            logger.info("Test set not found; using 'val' dataset for evaluation.")
        else:
            error_msg = "No evaluation dataset found; expected key 'test' or 'val' in data."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Accumulators for predictions and true labels
        all_preds: List[np.ndarray] = []
        all_true: List[np.ndarray] = []

        # Disable gradient computations for evaluation.
        with torch.no_grad():
            for batch in eval_loader:
                # Process batch into inputs and labels.
                if isinstance(batch, (list, tuple)):
                    inputs, labels = batch[0], batch[1]
                    if isinstance(inputs, torch.Tensor):
                        inputs = inputs.to(self.device)
                    if isinstance(labels, torch.Tensor):
                        labels = labels.to(self.device)
                    forward_inputs = inputs
                elif isinstance(batch, dict):
                    # Assume keys include "labels" and others to be passed as inputs.
                    labels = batch.get("labels")
                    if isinstance(labels, torch.Tensor):
                        labels = labels.to(self.device)
                    # Prepare inputs excluding the "labels" key.
                    forward_inputs = {k: v.to(self.device) for k, v in batch.items() if k != "labels"}
                else:
                    logger.error("Unrecognized batch format during evaluation; skipping batch.")
                    continue

                # Forward pass: if inputs is dict, use keyword unpacking.
                if isinstance(forward_inputs, dict):
                    outputs = self.model(**forward_inputs)
                else:
                    outputs = self.model(forward_inputs)

                if self.domain in {"cv", "nlp"}:
                    # For CV/NLP: use argmax to obtain predicted class indices.
                    preds = torch.argmax(outputs, dim=1)
                    preds_np = preds.detach().cpu().numpy()
                elif self.domain == "graph":
                    # For Graph: obtain probability estimates for the positive class.
                    if self.graph_apply_sigmoid:
                        if outputs.dim() == 2 and outputs.size(1) == 1:
                            # Binary output as single logit value.
                            probs = torch.sigmoid(outputs).view(-1)
                        elif outputs.dim() == 2 and outputs.size(1) == 2:
                            # Two class output: use softmax and take positive class probability.
                            probs = torch.softmax(outputs, dim=1)[:, 1]
                        else:
                            # Fallback: if unexpected shape, attempt softmax over last dimension.
                            probs = torch.softmax(outputs, dim=1)[:, -1]
                        preds_np = probs.detach().cpu().numpy()
                    else:
                        # Assume outputs are already probabilities.
                        if outputs.dim() == 2 and outputs.size(1) >= 2:
                            preds_np = outputs[:, 1].detach().cpu().numpy()
                        else:
                            preds_np = outputs.view(-1).detach().cpu().numpy()
                else:
                    logger.error(f"Unsupported domain '{self.domain}' during evaluation.")
                    raise ValueError(f"Unsupported domain '{self.domain}'.")

                # Accumulate true labels
                true_np = labels.detach().cpu().numpy()

                all_preds.append(preds_np)
                all_true.append(true_np)

        # Concatenate results from all batches.
        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_true)

        results: Dict[str, Any] = {}
        # Compute metrics based on domain
        if self.domain in {"cv", "nlp"}:
            # For CV and NLP, y_pred are discrete predicted class labels.
            if "accuracy" in [metric.lower() for metric in self.metrics]:
                acc = accuracy_score(y_true, y_pred)
                results["accuracy"] = acc
            if self.domain == "nlp":
                # Compute additional NLP metrics if specified.
                if any(metric.lower() == "f1" for metric in self.metrics):
                    f1 = f1_score(y_true, y_pred, average="macro")
                    results["F1"] = f1
                if any(metric.lower() == "spearman" for metric in self.metrics):
                    try:
                        spearman_corr, _ = spearmanr(y_true, y_pred)
                    except Exception as e:
                        logger.error(f"Error computing Spearman correlation: {e}")
                        spearman_corr = float('nan')
                    results["Spearman"] = spearman_corr
                if any(metric.lower() == "matthews" for metric in self.metrics):
                    matthews = matthews_corrcoef(y_true, y_pred)
                    results["Matthews"] = matthews
        elif self.domain == "graph":
            # For Graph tasks, compute AUC using continuous probability estimates.
            if any(metric.lower() == "auc" for metric in self.metrics):
                try:
                    auc = roc_auc_score(y_true, y_pred)
                except Exception as e:
                    logger.error(f"Error computing AUC: {e}")
                    auc = float('nan')
                results["AUC"] = auc

        # Optionally record evaluation time and GPU memory usage.
        if self.record_eval_time:
            eval_duration = time.time() - eval_start_time
            results["evaluation_time_sec"] = eval_duration
            logger.info(f"Evaluation time: {eval_duration:.2f} seconds")
        if self.record_gpu_memory and self.device.type == "cuda":
            gpu_memory_bytes = torch.cuda.max_memory_allocated(self.device)
            gpu_memory_gb = gpu_memory_bytes / (1024 ** 3)
            results["evaluation_gpu_memory_gb"] = gpu_memory_gb
            # Reset peak memory for subsequent runs.
            torch.cuda.reset_peak_memory_stats(self.device)
            logger.info(f"Evaluation GPU memory usage: {gpu_memory_gb:.2f} GB")

        logger.info(f"Evaluation results: {results}")
        return results
