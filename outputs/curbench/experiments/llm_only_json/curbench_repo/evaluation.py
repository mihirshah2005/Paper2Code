"""evaluation.py

This module defines the Evaluation class for performing inference on 
the test (or validation) dataset and computing performance metrics as
described in the CurBench benchmark paper. The Evaluation class explicitly
determines the training domain (cv, nlp, or graph) either from the provided
data dictionary or from the configuration, and processes the model outputs
based on an explicitly defined 'model_output_type' flag in the evaluation configuration.
It then computes domain‐specific metrics such as accuracy (CV), a suite of metrics
(accuracy, F1, Spearman, and Matthews for NLP), or AUC (graph) using scikit‑learn's
functions. Complexity metrics (e.g., training time and GPU memory usage) are added
if indicated in configuration.

Usage:
    evaluation_config = Config().get_config()
    evaluator = Evaluation(model, data_dict, evaluation_config)
    results = evaluator.evaluate()
    print(results)
"""

import logging
import torch
import numpy as np
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score

try:
    from scipy.stats import spearmanr
except ImportError:
    spearmanr = None
    logging.warning("scipy.stats.spearmanr is not available; Spearman correlation will not be computed.")

# Set up module-level logger.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class Evaluation:
    """
    Evaluation class performs inference and computes evaluation metrics
    for the CurBench experiments.

    __init__:
        - Determines the evaluation domain explicitly. It first checks 
          if the provided data dictionary contains a 'domain' key. If not, it 
          looks for 'domain' in the evaluation section of the configuration.
          If neither is provided, it raises a ValueError.
        - Retrieves the 'model_output_type' flag from configuration (defaulting to 'logits')
          to control the post-processing of model outputs.
        - Sets the computation device (GPU if available, otherwise CPU).

    evaluate():
        - Sets the model to evaluation mode and performs inference 
          over the test (or validation) dataset.
        - Depending on the domain and model_output_type, it post-processes outputs:
            • For "logits" in CV and NLP: applies torch.argmax for class prediction.
            • For "logits" in graph: applies sigmoid (or softmax if 2 outputs) to obtain probability scores.
            • For "probabilities": assumes outputs are already probabilities, or applies argmax for CV/NLP.
        - Accumulates predictions and true labels and computes domain-specific metrics:
            • CV: Accuracy (via sklearn.metrics.accuracy_score)
            • NLP: Accuracy, F1 Score (with binary or weighted averaging), Spearman correlation, and Matthews correlation.
            • Graph: ROC AUC (via sklearn.metrics.roc_auc_score)
        - Optionally, if configured, includes complexity metrics (training_time and gpu_memory)
          if they are provided in the data dictionary.
        - Returns a summary dictionary with the computed metrics and additional information.
    """

    def __init__(self, model: torch.nn.Module, data: dict, config: dict) -> None:
        """
        Initialize Evaluation with model, data dictionary, and configuration.

        Args:
            model (torch.nn.Module): The trained backbone model.
            data (dict): Dictionary containing at least the "test" dataset.
                         Optionally, it can contain a "domain" key and complexity metrics.
            config (dict): Configuration dictionary loaded from config.yaml.

        Raises:
            ValueError: If domain information is not provided in data or config.
        """
        self.model = model
        self.data = data
        self.config = config

        # Domain determination:
        self.domain = None
        if "domain" in self.data:
            self.domain = self.data["domain"]
        else:
            eval_config = self.config.get("evaluation", {})
            self.domain = eval_config.get("domain", None)
        if self.domain is None:
            raise ValueError("Domain information missing. Please specify a 'domain' key in the data "
                             "dictionary or in the configuration under evaluation.domain.")
        self.domain = self.domain.lower()
        logger.info(f"Evaluation domain set to '{self.domain}'.")

        # Determine the model output type:
        self.model_output_type = self.config.get("evaluation", {}).get("model_output_type", "logits")
        if "model_output_type" not in self.config.get("evaluation", {}):
            logger.warning("Evaluation: 'model_output_type' not found in configuration; defaulting to 'logits'.")
        logger.info(f"Model output type set to '{self.model_output_type}'.")

        # Set computation device.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def evaluate(self) -> dict:
        """
        Evaluates the model on the test dataset and computes domain-specific metrics.

        Returns:
            dict: Summary of computed evaluation metrics and, if configured, complexity metrics.
        """
        self.model.eval()
        all_preds = []
        all_labels = []

        # Determine test DataLoader:
        test_data = self.data.get("test", None)
        if test_data is None:
            raise KeyError("Test dataset ('test' key) not found in the provided data dictionary.")

        if isinstance(test_data, DataLoader):
            test_loader = test_data
        else:
            # Use a default batch size of 50 if not provided.
            test_loader = DataLoader(test_data, batch_size=50, shuffle=False)

        with torch.no_grad():
            for batch in test_loader:
                # Domain-specific data extraction:
                if self.domain in ["cv", "nlp"]:
                    try:
                        inputs, labels = batch
                    except Exception as e:
                        logger.error(f"Error unpacking batch for domain '{self.domain}': {e}")
                        continue
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                elif self.domain == "graph":
                    # For graph data, assume batch is a torch_geometric Batch object.
                    batch = batch.to(self.device)
                    inputs = batch
                    if not hasattr(batch, "y"):
                        logger.error("Graph batch does not contain attribute 'y' for labels.")
                        continue
                    labels = batch.y.to(self.device)
                else:
                    raise ValueError(f"Unsupported domain '{self.domain}' encountered during evaluation.")

                # Forward pass:
                try:
                    outputs = self.model(inputs)
                except Exception as e:
                    logger.error(f"Error during model inference: {e}")
                    continue

                # Process outputs based on model_output_type:
                if self.model_output_type.lower() == "logits":
                    if self.domain in ["cv", "nlp"]:
                        preds = torch.argmax(outputs, dim=1)
                    elif self.domain == "graph":
                        if outputs.dim() == 2:
                            if outputs.size(1) == 1:
                                preds = torch.sigmoid(outputs).squeeze(1)
                            elif outputs.size(1) == 2:
                                probabilities = torch.softmax(outputs, dim=1)
                                preds = probabilities[:, 1]
                            else:
                                preds = torch.argmax(outputs, dim=1)
                        else:
                            preds = torch.sigmoid(outputs)
                    else:
                        raise ValueError(f"Unsupported domain '{self.domain}' for logits processing.")
                elif self.model_output_type.lower() == "probabilities":
                    if self.domain in ["cv", "nlp"]:
                        preds = torch.argmax(outputs, dim=1)
                    elif self.domain == "graph":
                        if outputs.dim() == 2 and outputs.size(1) == 2:
                            preds = outputs[:, 1]
                        else:
                            preds = outputs
                    else:
                        raise ValueError(f"Unsupported domain '{self.domain}' for probabilities processing.")
                else:
                    logger.warning("Unrecognized 'model_output_type'; defaulting to 'logits' processing.")
                    if self.domain in ["cv", "nlp"]:
                        preds = torch.argmax(outputs, dim=1)
                    elif self.domain == "graph":
                        preds = torch.sigmoid(outputs)

                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        # Concatenate predictions and labels.
        try:
            preds_tensor = torch.cat(all_preds)
        except Exception as e:
            logger.error(f"Error concatenating predictions: {e}")
            preds_tensor = torch.tensor([])
        try:
            labels_tensor = torch.cat(all_labels)
        except Exception as e:
            logger.error(f"Error concatenating labels: {e}")
            labels_tensor = torch.tensor([])

        y_pred = preds_tensor.numpy()
        y_true = labels_tensor.numpy()

        summary = {"domain": self.domain}

        # Domain-specific metric computation:
        if self.domain == "cv":
            # Compute accuracy.
            acc = accuracy_score(y_true, y_pred)
            summary["accuracy"] = acc
        elif self.domain == "nlp":
            # Retrieve evaluation metrics list from configuration.
            eval_metrics = self.config.get("evaluation", {}).get("metrics", [])
            if not isinstance(eval_metrics, list):
                eval_metrics = [eval_metrics]

            # Accuracy:
            acc = accuracy_score(y_true, y_pred)
            summary["accuracy"] = acc

            # F1 Score: determine average method based on binary vs multi-class.
            unique_labels = np.unique(y_true)
            avg_method = "binary" if len(unique_labels) == 2 else "weighted"
            f1 = f1_score(y_true, y_pred, average=avg_method)
            summary["F1"] = f1

            # Spearman correlation:
            if spearmanr is not None:
                try:
                    spearman_corr, _ = spearmanr(y_true, y_pred)
                    summary["Spearman"] = spearman_corr
                except Exception as e:
                    logger.error(f"Error computing Spearman correlation: {e}")
                    summary["Spearman"] = None
            else:
                summary["Spearman"] = None
                logger.warning("scipy.stats.spearmanr not available; skipping Spearman correlation.")

            # Matthews correlation coefficient:
            matthews = matthews_corrcoef(y_true, y_pred)
            summary["Matthews"] = matthews
        elif self.domain == "graph":
            # Compute ROC AUC score for binary classification.
            try:
                auc = roc_auc_score(y_true, y_pred)
                summary["AUC"] = auc
            except Exception as e:
                logger.error(f"Error computing ROC AUC: {e}")
                summary["AUC"] = None
        else:
            raise ValueError(f"Unsupported domain '{self.domain}' for metric computation.")

        # Incorporate complexity metrics if configured.
        complexity_config = self.config.get("evaluation", {}).get("record_complexity", {})
        if complexity_config.get("training_time", False):
            summary["training_time"] = self.data.get("training_time", None)
        if complexity_config.get("gpu_memory", False):
            summary["gpu_memory"] = self.data.get("gpu_memory", None)

        return summary
