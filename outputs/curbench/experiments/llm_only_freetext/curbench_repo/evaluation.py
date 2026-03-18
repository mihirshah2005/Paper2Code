"""evaluation.py

This module defines the Evaluation class which performs inference on a provided dataset,
computes performance metrics (accuracy for CV; accuracy, F1-score, Spearman correlation, 
and Matthews correlation for NLP; AUC for Graph), and optionally records complexity metrics 
(inference time and peak GPU memory usage) based on the configuration settings.

The Evaluation class is initialized with:
 • model: the trained backbone model (an instance of nn.Module)
 • data: a dictionary containing datasets; uses "test" if available, otherwise "val"
 • config: configuration dictionary loaded via config.py (e.g., from config.yaml)
 
Usage:
    evaluator = Evaluation(model, data, config)
    summary = evaluator.evaluate()
    print(summary)
"""

import time
import logging
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from scipy.stats import spearmanr
import numpy as np

# Set up module-level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class Evaluation:
    """
    Evaluation class performs inference on the provided dataset using the trained model,
    computes domain-specific performance metrics, and optionally records complexity metrics.
    
    Attributes:
        model (nn.Module): The trained model.
        data (dict): A dictionary containing dataset splits ("train", "val", "test").
        config (dict): Configuration dictionary containing evaluation parameters.
        domain (str): Evaluation domain: "cv", "nlp", or "graph", inferred from provided data.
        device (torch.device): Computation device ("cuda" if available, else "cpu").
    """
    
    def __init__(self, model: nn.Module, data: Dict[str, Any], config: Dict[str, Any]) -> None:
        """
        Initializes the Evaluation class.
        
        :param model: The trained neural network model.
        :param data: Dataset dictionary. Must contain at least one of "test" or "val" keys.
        :param config: Configuration dictionary from config.yaml.
        """
        self.model = model
        self.data = data
        self.config = config
        
        # Determine the evaluation domain based on the data characteristics.
        self.domain = self._detect_domain(self.data)
        logger.info(f"Evaluation domain set to '{self.domain}'.")
        
        # Set device from model parameters.
        self.device = next(self.model.parameters()).device
        logger.info(f"Evaluation will run on device: {self.device}.")

    def _detect_domain(self, data: Dict[str, Any]) -> str:
        """
        Detects the domain ("cv", "nlp", or "graph") from the provided dataset.
        The following heuristics are used:
            - If the dataset (test/val) is a list and its first element is a dict containing a "label" key, domain is assumed to be "nlp".
            - If the dataset has an attribute "targets", domain is assumed to be "cv".
            - If the dataset (or its first element within a Subset) has an attribute "x" (typical for graph data), domain is "graph".
            - Otherwise, defaults to "cv".
        
        :param data: Dictionary containing dataset splits.
        :return: Domain string.
        """
        eval_dataset = None
        if "test" in data:
            eval_dataset = data["test"]
            loader_key = "test"
        elif "val" in data:
            eval_dataset = data["val"]
            loader_key = "val"
        else:
            error_msg = "No valid evaluation dataset provided. Expected 'test' or 'val' in data dictionary."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # If eval_dataset is a DataLoader, try to extract a sample from its dataset.
        sample = None
        try:
            if isinstance(eval_dataset, DataLoader):
                for batch in eval_dataset:
                    sample = batch
                    break
            elif isinstance(eval_dataset, (Dataset, Subset)):
                sample = eval_dataset[0]
            elif isinstance(eval_dataset, list):
                sample = eval_dataset[0]
            else:
                logger.warning("Unable to determine dataset type; defaulting to 'cv'.")
                return "cv"
        except Exception as e:
            logger.warning(f"Error extracting sample from evaluation data: {e}. Defaulting to 'cv'.")
            return "cv"
        
        # Heuristics based on sample type.
        if isinstance(sample, dict):
            # For NLP, expect a dict with a "label" key.
            if "label" in sample:
                return "nlp"
        elif isinstance(sample, (list, tuple)):
            # Assume tuple/list containing (input, label)
            # In CV, typically the second element holds the label.
            if len(sample) >= 2:
                # Try to check if label is a tensor or numeric.
                label = sample[1]
                if torch.is_tensor(label) or isinstance(label, (int, np.integer)):
                    return "cv"
        else:
            # Check for attribute "targets" (common in torchvision datasets)
            if hasattr(sample, "targets"):
                return "cv"
            # For graph datasets, sample might be an object with attribute "x"
            if hasattr(sample, "x"):
                return "graph"
        # Default fallback.
        return "cv"

    def _prepare_dataloader(self, dataset: Any) -> DataLoader:
        """
        Prepares a DataLoader from the given dataset if it is not already a DataLoader.
        Sets the default batch_size to 50 and no shuffling.
        
        :param dataset: The dataset object.
        :return: DataLoader instance.
        """
        if isinstance(dataset, DataLoader):
            return dataset
        else:
            # Default batch size for evaluation.
            default_batch_size = 50
            return DataLoader(dataset, batch_size=default_batch_size, shuffle=False, num_workers=0)

    def _unpack_batch(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Unpacks a batch into inputs and labels.
        
        Handles both tuples/lists and dictionaries.
        
        :param batch: A batch loaded from DataLoader.
        :return: Tuple (inputs, labels) moved to the evaluation device.
        :raises: ValueError if batch format is unrecognized.
        """
        if isinstance(batch, (list, tuple)):
            inputs, labels = batch[0], batch[1]
        elif isinstance(batch, dict):
            if "input_ids" in batch and "labels" in batch:
                inputs, labels = batch["input_ids"], batch["labels"]
            elif "data" in batch and "target" in batch:
                inputs, labels = batch["data"], batch["target"]
            elif "image" in batch and "label" in batch:
                inputs, labels = batch["image"], batch["label"]
            else:
                logger.error("Unrecognized batch dictionary format for evaluation.")
                raise ValueError("Batch dictionary does not contain recognized keys.")
        else:
            logger.error("Unrecognized batch format; expected tuple, list or dict.")
            raise ValueError("Unable to unpack batch data.")
        
        # Move tensors to device
        if torch.is_tensor(inputs):
            inputs = inputs.to(self.device)
        if torch.is_tensor(labels):
            labels = labels.to(self.device)
        return inputs, labels

    def evaluate(self) -> Dict[str, Any]:
        """
        Runs evaluation on the provided dataset (prefers "test" split, falls back to "val") 
        and computes domain-specific performance metrics along with optional complexity metrics.

        Steps:
             1. Set model to evaluation mode.
             2. Select evaluation dataset (test > val; error if neither exists).
             3. Reset GPU memory counters (if applicable) and start inference timer.
             4. Iterate over batches (in torch.no_grad) and obtain predictions and true labels.
             5. Apply domain-specific post-processing:
                  - For CV and NLP: use argmax over logits.
                  - For Graph: apply sigmoid to obtain probabilities.
             6. Compute performance metrics:
                  - CV: accuracy.
                  - NLP: accuracy, F1 score, Spearman correlation, Matthews correlation.
                  - Graph: AUC.
             7. Record complexity metrics if enabled: inference time and peak GPU memory usage.
             8. Return a summary dictionary.
             
        :return: Summary dictionary with "performance" and "complexity" keys.
        """
        # Step A: Set model to evaluation mode.
        self.model.eval()
        
        # Step B: Select the evaluation dataset.
        if "test" in self.data:
            eval_dataset = self.data["test"]
            logger.info("Using 'test' dataset for evaluation.")
        elif "val" in self.data:
            eval_dataset = self.data["val"]
            logger.info("Using 'val' dataset for evaluation as 'test' is not provided.")
        else:
            error_msg = "No evaluation dataset available. Provide either 'test' or 'val' in data dictionary."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        eval_loader = self._prepare_dataloader(eval_dataset)
        
        # Step C: Reset GPU memory counters (if using CUDA) and start timer.
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
        inference_start_time = time.time()
        
        all_preds: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []
        
        # Step D: Inference loop in evaluation mode.
        with torch.no_grad():
            for batch in eval_loader:
                try:
                    inputs, labels = self._unpack_batch(batch)
                except Exception as e:
                    logger.error(f"Error unpacking batch during evaluation: {e}")
                    continue

                outputs = self.model(inputs)
                
                # Domain-specific post-processing
                if self.domain in ["cv", "nlp"]:
                    # Assume outputs are logits; take argmax along dim=1.
                    preds = torch.argmax(outputs, dim=1)
                elif self.domain == "graph":
                    # For graph tasks, apply sigmoid to obtain probabilities.
                    preds = torch.sigmoid(outputs)
                    # For AUC, predictions as probability scores are required.
                    # If outputs have more than one column, assume binary classification with second column probability.
                    if preds.dim() > 1 and preds.size(1) > 1:
                        preds = preds[:, 1]
                    else:
                        preds = preds.squeeze()  # Ensure it's 1D
                else:
                    logger.warning("Unrecognized domain for evaluation post-processing; defaulting to argmax.")
                    preds = torch.argmax(outputs, dim=1)
                
                # Move predictions and labels to CPU and convert to numpy arrays.
                preds_np = preds.detach().cpu().numpy()
                labels_np = labels.detach().cpu().numpy()
                
                all_preds.append(preds_np)
                all_labels.append(labels_np)
        
        # Concatenate arrays from all batches.
        if all_preds:
            predictions = np.concatenate(all_preds, axis=0)
            true_labels = np.concatenate(all_labels, axis=0)
        else:
            error_msg = "No predictions were made during evaluation (empty dataset?)."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Step E: Record complexity metrics.
        inference_time = time.time() - inference_start_time
        if torch.cuda.is_available():
            peak_gpu_memory = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)  # in GB
        else:
            peak_gpu_memory = 0.0
        
        # Step F: Domain-Specific Metrics Computation.
        performance_metrics: Dict[str, Any] = {}
        
        if self.domain == "cv":
            # For computer vision tasks, compute accuracy.
            acc = accuracy_score(true_labels, predictions)
            performance_metrics["accuracy"] = acc
            
        elif self.domain == "nlp":
            # For NLP, compute multiple metrics.
            acc = accuracy_score(true_labels, predictions)
            f1 = f1_score(true_labels, predictions, average="macro")
            # Spearman correlation: if predictions and true labels are continuous or ranking, here we use them directly.
            # Note: Spearman expects the two inputs to be of the same shape.
            spearman_corr, _ = spearmanr(true_labels, predictions)
            matthews = matthews_corrcoef(true_labels, predictions)
            performance_metrics["accuracy"] = acc
            performance_metrics["f1_score"] = f1
            performance_metrics["spearman"] = spearman_corr
            performance_metrics["matthews_corrcoef"] = matthews

        elif self.domain == "graph":
            # For graph tasks, compute AUC.
            # Ensure that true_labels are binary (0 or 1) and predictions are probability scores.
            try:
                auc = roc_auc_score(true_labels, predictions)
                performance_metrics["auc"] = auc
            except Exception as e:
                logger.error(f"Error computing AUC: {e}")
                performance_metrics["auc"] = None
        else:
            logger.warning("Domain not recognized for metric computation; no performance metrics computed.")
        
        # Step G: Prepare complexity metrics based on configuration.
        record_comp = self.config.get("evaluation", {}).get("record_complexity", {})
        complexity_metrics: Dict[str, Any] = {}
        if record_comp.get("training_time", False):
            complexity_metrics["inference_time_sec"] = inference_time
        if record_comp.get("gpu_memory", False):
            complexity_metrics["peak_gpu_memory_gb"] = peak_gpu_memory
        
        summary: Dict[str, Any] = {
            "performance": performance_metrics,
            "complexity": complexity_metrics
        }
        
        logger.info(f"Evaluation completed. Performance metrics: {performance_metrics}")
        if complexity_metrics:
            logger.info(f"Complexity metrics: {complexity_metrics}")
        
        return summary
