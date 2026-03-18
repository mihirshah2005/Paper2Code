"""
evaluation.py

This module implements the Evaluation class which defines the evaluation logic for the 
Curriculum Learning Benchmark (CurBench). It supports domain-specific metric computation,
including Accuracy for CV, a suite of metrics (Accuracy, F1, Spearman correlation, Matthews correlation)
for NLP, and AUC for Graph tasks. The Evaluation class accepts a trained model, a dictionary of data loaders,
and the experiment configuration (loaded from config.yaml) to compute a summary dictionary of evaluation metrics.
"""

import logging
import torch
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score

# Setup basic logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """
    Computes the Spearman correlation coefficient between two arrays.
    This is done by ranking both arrays and then computing the Pearson correlation coefficient
    between the ranked arrays.

    Args:
        x (np.ndarray): Predictions as a 1D numpy array.
        y (np.ndarray): Ground truth labels as a 1D numpy array.

    Returns:
        float: Spearman correlation coefficient.
    """
    # Obtain the ranks by applying argsort twice
    rank_x = np.argsort(np.argsort(x))
    rank_y = np.argsort(np.argsort(y))
    # Compute Pearson correlation on the rank arrays
    if rank_x.size < 2:
        return 0.0
    corr_matrix = np.corrcoef(rank_x, rank_y)
    return float(corr_matrix[0, 1])


class Evaluation:
    """
    The Evaluation class computes domain-specific evaluation metrics for a trained model.
    
    Attributes:
        model (torch.nn.Module): The trained model to be evaluated.
        data (dict): Dictionary containing evaluation data loaders with keys such as "train", "val", "test".
        config (dict): Experiment configuration dictionary (from config.yaml) containing metric settings.
        device (torch.device): Computation device determined from the model parameters.
        domain (str): Inferred domain identifier ("cv", "nlp", or "graph") based on the model's type.
    
    Methods:
        evaluate() -> dict:
            Runs the evaluation loop on the selected data loader and computes metrics
            based on the domain. Returns a summary dictionary of computed metrics.
    """

    def __init__(self, model: torch.nn.Module, data: dict, config: dict) -> None:
        """
        Initializes the Evaluation instance.

        Args:
            model (torch.nn.Module): The trained model instance.
            data (dict): A dictionary containing the evaluation datasets (data loaders) with keys such as "test" or "val".
            config (dict): The experiment configuration dictionary (parsed from config.yaml).
        """
        self.model = model
        self.data = data
        self.config = config
        self.device = self._get_device()
        # Determine the domain based on model class name
        model_class_name = self.model.__class__.__name__.lower()
        if any(sub in model_class_name for sub in ["lenet", "resnet", "vit"]):
            self.domain = "cv"
        elif any(sub in model_class_name for sub in ["lstm", "bert", "gpt2"]):
            self.domain = "nlp"
        elif any(sub in model_class_name for sub in ["gcn", "gat", "gin"]):
            self.domain = "graph"
        else:
            self.domain = "cv"  # Default to CV if not recognized
        logger.info("Evaluation initialized for domain '%s' on device %s.", self.domain, self.device)

    def _get_device(self) -> torch.device:
        """
        Determines the device of the model based on its parameters.

        Returns:
            torch.device: The device (GPU if available, else CPU).
        """
        return next(self.model.parameters()).device

    def _to_device(self, data_item: any, device: torch.device) -> any:
        """
        Recursively moves data items (tensors, lists, dicts) to the specified device.

        Args:
            data_item: A tensor, list, tuple, or dict containing tensors.
            device (torch.device): The target device.

        Returns:
            The data item with all torch.Tensor elements moved to the specified device.
        """
        if isinstance(data_item, torch.Tensor):
            return data_item.to(device)
        elif isinstance(data_item, (list, tuple)):
            return type(data_item)(self._to_device(item, device) for item in data_item)
        elif isinstance(data_item, dict):
            return {key: self._to_device(value, device) for key, value in data_item.items()}
        else:
            return data_item

    def evaluate(self) -> dict:
        """
        Evaluates the trained model on the appropriate dataset (test set for CV/Graph, validation for NLP)
        and computes domain-specific metrics.

        The evaluation loop disables gradient computation, collects predictions and ground-truth labels,
        and then computes:
            - For CV: Accuracy.
            - For NLP: Accuracy, F1 (macro), Matthews correlation, and Spearman correlation.
            - For Graph: ROC AUC (using predicted probabilities for the positive class).

        Returns:
            dict: A summary dictionary containing the computed evaluation metrics.
        """
        self.model.eval()
        all_preds = []
        all_labels = []

        # Select evaluation data loader:
        # For NLP, always use 'val' as test labels are unavailable.
        # For CV and Graph, use 'test' if available; otherwise, fallback to 'val'.
        if self.domain == "nlp":
            eval_loader = self.data.get("val")
        else:
            eval_loader = self.data.get("test", self.data.get("val"))
        if eval_loader is None:
            raise ValueError("No evaluation data loader found in the provided data dictionary (expecting 'test' or 'val').")

        with torch.no_grad():
            for batch in eval_loader:
                # Unpack the batch based on its type.
                if isinstance(batch, (list, tuple)):
                    if len(batch) < 2:
                        raise ValueError("Expected batch tuple to contain (inputs, labels).")
                    inputs = batch[0]
                    labels = batch[1]
                elif isinstance(batch, dict):
                    if "label" in batch:
                        labels = batch["label"]
                    elif "labels" in batch:
                        labels = batch["labels"]
                    else:
                        raise ValueError("Label key ('label' or 'labels') not found in batch dictionary.")
                    inputs = batch
                else:
                    # Assume graph data object with attribute 'y' for labels.
                    inputs = batch
                    if hasattr(batch, "y"):
                        labels = batch.y
                    else:
                        raise ValueError("Graph batch object does not have attribute 'y' for labels.")

                # Move inputs and labels to the same device as the model.
                inputs = self._to_device(inputs, self.device)
                labels = self._to_device(labels, self.device)

                # Forward pass through the model.
                outputs = self.model(inputs)
                # Handle models that output a wrapper with logits.
                if hasattr(outputs, "logits"):
                    logits = outputs.logits
                else:
                    logits = outputs

                # Post-process outputs based on domain.
                if self.domain in ["cv", "nlp"]:
                    # For classification tasks, choose the class with highest logit.
                    preds = torch.argmax(logits, dim=1)
                elif self.domain == "graph":
                    # For binary graph classification tasks:
                    # If logits shape is [B, 2], apply softmax and take probability of class 1.
                    if logits.dim() == 2 and logits.size(1) == 2:
                        probs = torch.softmax(logits, dim=1)
                        preds = probs[:, 1]  # Continuous probabilities for positive class
                    else:
                        # If logits is one-dimensional, apply sigmoid.
                        probs = torch.sigmoid(logits)
                        preds = probs.squeeze()
                else:
                    preds = torch.argmax(logits, dim=1)

                # Accumulate predictions and ground truth labels.
                all_preds.extend(preds.detach().cpu().numpy())
                all_labels.extend(labels.detach().cpu().numpy())

        # Convert lists to NumPy arrays for metric computation.
        np_preds = np.array(all_preds)
        np_labels = np.array(all_labels)
        metrics_result = {}

        # Compute domain-specific metrics.
        if self.domain == "cv":
            # Compute accuracy for computer vision tasks.
            acc = accuracy_score(np_labels, np_preds)
            metrics_result["accuracy"] = acc

        elif self.domain == "nlp":
            # For NLP tasks, compute multiple metrics.
            acc = accuracy_score(np_labels, np_preds)
            f1 = f1_score(np_labels, np_preds, average="macro")
            mcc = matthews_corrcoef(np_labels, np_preds)
            spearman = _spearman_correlation(np_preds, np_labels)
            metrics_result["accuracy"] = acc
            metrics_result["F1"] = f1
            metrics_result["Matthews"] = mcc
            metrics_result["Spearman"] = spearman

        elif self.domain == "graph":
            # For Graph tasks (binary classification), compute ROC AUC.
            try:
                auc = roc_auc_score(np_labels, np_preds)
            except Exception as e:
                logger.error("Error computing ROC AUC: %s", e)
                auc = None
            metrics_result["AUC"] = auc

        else:
            # Default: compute accuracy.
            acc = accuracy_score(np_labels, np_preds)
            metrics_result["accuracy"] = acc

        logger.info("Evaluation metrics computed: %s", metrics_result)
        return metrics_result
