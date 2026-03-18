"""
evaluation.py

This module implements the Evaluation class for assessing both classifier
performance (ACC and ASR) and detection performance (TPR and FPR) on a test dataset.
Each sample in the test dataset is expected to be a dictionary containing:
  - "image": either a PIL Image or a preprocessed torch.Tensor;
  - "label": an integer representing the ground truth class label;
  - Optionally "is_dirty": a boolean flag that denotes the ground truth dirty status;
      if absent, it will be computed from the "meta" dictionary (using keys "trigger_injected" or "noisy_injected").
  - Optionally "detected_dirty": the detector output; if numeric then a lower value (below threshold)
      indicates the sample is considered dirty; if Boolean, then it is used directly.
If "detected_dirty" is missing, a fallback key "is_poisoned" is used. If no detection field is found,
detection metrics will be skipped for that sample.

The Evaluation class uses a DataLoader (with a batch size configured from config.yaml) to iterate
over the test dataset, feeds the images to the provided classifier (ResNet-18), and then computes:
  - ACC: Clean sample accuracy (for samples with ground-truth dirty flag False).
  - ASR: Attack Success Rate (for dirty samples, i.e. ground-truth True) defined as the fraction
         of dirty samples predicted as the attacker’s target label.
  - TPR (True Positive Rate) and FPR (False Positive Rate) for detection, by thresholding raw detection
         scores (if available) with the detection threshold (default 0.5 from config).

Configuration parameters (e.g., batch size, detection threshold, target label) are obtained from config.yaml.
Default values are always set if not provided.

Dependencies:
  - numpy==1.21.0
  - torch==1.9.0
  - torchvision==0.10.0
  - transformers==4.15.0
  - openai==0.27.0
  - requests==2.26.0
  - PyYAML (for YAML configuration parsing)

Author: Your Name
Date: YYYY-MM-DD
"""

import os
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset, DataLoader

# Import utilities for configuration, seed initialization, and image preprocessing.
from utils import load_config, init_seed, preprocess_image

# Set up logger for the evaluation module.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class EvaluationDataset(Dataset):
    """
    A custom PyTorch Dataset that wraps a list of sample dictionaries for evaluation.
    Each sample is expected to be a dictionary containing:
      - "image": a PIL Image or torch.Tensor.
      - "label": an integer, the ground truth label.
      - Optionally "is_dirty": boolean ground truth dirty flag.
      - Optionally "meta": a dictionary containing flags like "trigger_injected" and "noisy_injected".
      - Optionally "detected_dirty": the detector output from detection module.
    If "is_dirty" is not provided, it will be computed from meta.
    """
    def __init__(self, samples: List[Dict[str, Any]], dataset_name: str, transform: Optional[Any] = None) -> None:
        """
        Initializes the EvaluationDataset.

        Args:
            samples (List[Dict[str, Any]]): List of sample dictionaries.
            dataset_name (str): Name of the dataset ("CIFAR-10", "ImageNet-100", or "ImageNet-Dog").
            transform: Optional transform function to preprocess PIL Images into torch.Tensors.
                       If not provided, a default transform using preprocess_image() is applied.
        """
        super().__init__()
        self.samples = samples
        self.dataset_name = dataset_name
        if transform is not None:
            self.transform = transform
        else:
            # Default transformation using the provided utility function.
            self.transform = lambda img: preprocess_image(img, self.dataset_name)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, bool, Optional[Union[bool, float, int]]]:
        sample = self.samples[index]
        image = sample.get("image")
        # If image is not a tensor, apply transformation.
        if not isinstance(image, torch.Tensor):
            image = self.transform(image)
        label = sample.get("label", -1)
        # Determine ground truth dirty flag:
        if "is_dirty" in sample:
            ground_truth_dirty = bool(sample["is_dirty"])
        elif "meta" in sample:
            meta = sample["meta"]
            ground_truth_dirty = bool(meta.get("trigger_injected", False) or meta.get("noisy_injected", False))
        else:
            ground_truth_dirty = False  # Default to clean if no flag is available.
        # Retrieve detection output. First check "detected_dirty", then fallback to "is_poisoned".
        if "detected_dirty" in sample:
            detection_out = sample["detected_dirty"]
        elif "is_poisoned" in sample:
            detection_out = sample["is_poisoned"]
        else:
            detection_out = None
        return image, label, ground_truth_dirty, detection_out


class Evaluation:
    """
    Evaluation class for computing classifier and detection performance metrics.
    
    It computes:
      - ACC (clean sample accuracy): For samples that are clean, compares model prediction with ground truth.
      - ASR (attack success rate): For dirty samples, calculates the fraction that are predicted as 
          the attacker’s target label.
      - TPR (True Positive Rate): For detection; among truly dirty samples, the fraction correctly flagged.
      - FPR (False Positive Rate): For detection; among truly clean samples, the fraction falsely flagged.
    
    The Evaluation class wraps the test dataset in a DataLoader, using batch size from the configuration.
    The target label for backdoor attack is read from configuration (default is 0).
    The detection threshold (default 0.5) is used to interpret numeric detection scores
    (a score < threshold is interpreted as a detection of "dirty").
    """
    
    def __init__(self, model: Any, test_data: Union[List[Dict[str, Any]], Dataset], config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initializes the Evaluation instance.

        Args:
            model (Any): The trained classifier model (e.g., ResNet-18).
            test_data (Union[List[Dict[str, Any]], Dataset]): The test dataset, either as a list of sample dicts
                    or a PyTorch Dataset.
            config (Optional[Dict[str, Any]]): Configuration dictionary loaded from config.yaml.
                                                If not provided, it will be loaded from "config.yaml".
        """
        # Load configuration from file if not provided.
        self.config: Dict[str, Any] = config if config is not None else load_config("config.yaml")
        # Set seed for reproducibility.
        seed_value: int = self.config.get("seed", 42)
        init_seed(seed_value)

        self.model = model
        self.model.eval()
        # Determine dataset name; try to obtain from config["training"]["dataset_name"], else default to "CIFAR-10"
        training_config: Dict[str, Any] = self.config.get("training", {})
        self.dataset_name: str = training_config.get("dataset_name", "CIFAR-10")

        # Set batch size from configuration under training->batch_size, using dataset_name as key.
        self.batch_size: int = training_config.get("batch_size", {}).get(self.dataset_name, 128)

        # Get detection threshold from config["detection"]["threshold"], default 0.5
        detection_config: Dict[str, Any] = self.config.get("detection", {})
        self.detection_threshold: float = float(detection_config.get("threshold", 0.5))

        # Get attacker target label from config["dirty_sample_generation"]["target_label"], default to 0.
        dirty_gen_config: Dict[str, Any] = self.config.get("dirty_sample_generation", {})
        self.target_label: int = int(dirty_gen_config.get("target_label", 0))
        
        # Wrap test_data into a PyTorch Dataset if it is a list.
        if isinstance(test_data, list):
            self.test_dataset = EvaluationDataset(test_data, self.dataset_name)
        elif isinstance(test_data, Dataset):
            self.test_dataset = test_data
        else:
            raise ValueError("test_data must be either a list of dictionaries or a PyTorch Dataset.")

        # Create a DataLoader for evaluation.
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
        logger.info(f"Evaluation initialized for dataset '{self.dataset_name}' with batch size {self.batch_size}, "
                    f"detection threshold {self.detection_threshold}, and target label {self.target_label}.")

    def evaluate(self) -> Dict[str, float]:
        """
        Runs evaluation on the test dataset.

        Computes:
          - ACC: Clean sample accuracy (for samples that are truly clean).
          - ASR: Attack Success Rate (for samples that are truly dirty,
                  evaluated as the fraction predicted as the target label).
          - Detection metrics: TPR (True Positive Rate) and FPR (False Positive Rate)
              based on the detector output, using fallback fields if necessary.

        Returns:
            Dict[str, float]: Dictionary containing the computed metrics.
                Keys include:
                    "ACC" - accuracy on clean samples,
                    "ASR" - attack success rate on dirty samples,
                    "TPR" - true positive rate for detection (if computable, else -1),
                    "FPR" - false positive rate for detection (if computable, else -1).
        """
        self.model.eval()
        total_clean: int = 0
        correct_clean: int = 0
        total_dirty: int = 0
        attack_success: int = 0

        # Counters for detection evaluation:
        true_positive: int = 0  # TP: ground truth dirty and detected as dirty.
        false_negative: int = 0  # FN: ground truth dirty but not detected.
        false_positive: int = 0  # FP: ground truth clean but detected as dirty.
        true_negative: int = 0   # TN: ground truth clean and not detected.
        detection_samples: int = 0  # Count of samples with available detection output.

        with torch.no_grad():
            for batch in self.test_loader:
                # Expect batch to be a tuple: (images, labels, ground_truth_dirty, detection_output)
                images, labels, dirty_flags, detection_outputs = batch
                images = images.to(next(self.model.parameters()).device)
                outputs = self.model(images)
                # Get predictions.
                _, predicted = torch.max(outputs, dim=1)

                batch_size_actual = labels.size(0)
                for i in range(batch_size_actual):
                    gt_label: int = int(labels[i].item())
                    pred_label: int = int(predicted[i].item())
                    is_dirty: bool = bool(dirty_flags[i].item()) if isinstance(dirty_flags[i], torch.Tensor) else bool(dirty_flags[i])
                    
                    # Classifier evaluation:
                    if not is_dirty:
                        total_clean += 1
                        if pred_label == gt_label:
                            correct_clean += 1
                    else:
                        total_dirty += 1
                        # For dirty samples, target label prediction counts as a successful attack.
                        if pred_label == self.target_label:
                            attack_success += 1

                    # Detection evaluation:
                    detected_value = detection_outputs[i]
                    detection_decision: Optional[bool] = None
                    if detected_value is not None:
                        # Check numeric type.
                        if isinstance(detected_value, (int, float)):
                            # For numeric detection scores, lower than threshold indicates detected dirty.
                            detection_decision = (float(detected_value) < self.detection_threshold)
                        elif isinstance(detected_value, bool):
                            detection_decision = detected_value
                        else:
                            logger.warning(f"Detection output type unrecognized: {detected_value} (type {type(detected_value)}). Skipping detection evaluation for this sample.")
                        if detection_decision is not None:
                            detection_samples += 1
                            if is_dirty:
                                if detection_decision:
                                    true_positive += 1
                                else:
                                    false_negative += 1
                            else:
                                if detection_decision:
                                    false_positive += 1
                                else:
                                    true_negative += 1
                    else:
                        logger.warning("No detection output found for a sample; skipping detection metric update for this sample.")

        # Compute classifier metrics.
        ACC: float = (correct_clean / total_clean) if total_clean > 0 else 0.0
        ASR: float = (attack_success / total_dirty) if total_dirty > 0 else 0.0

        # Compute detection metrics if at least one sample had detection output.
        TPR: float = -1.0
        FPR: float = -1.0
        if (true_positive + false_negative) > 0:
            TPR = true_positive / (true_positive + false_negative)
        else:
            logger.warning("No dirty samples with detection output available for TPR computation.")
        if (false_positive + true_negative) > 0:
            FPR = false_positive / (false_positive + true_negative)
        else:
            logger.warning("No clean samples with detection output available for FPR computation.")

        metrics: Dict[str, float] = {
            "ACC": ACC,
            "ASR": ASR,
            "TPR": TPR,
            "FPR": FPR
        }
        logger.info(f"Evaluation Metrics: ACC (Clean Accuracy) = {ACC * 100:.2f}%, ASR (Attack Success Rate) = {ASR * 100:.2f}%")
        if detection_samples > 0:
            logger.info(f"Detection Metrics: TPR (True Positive Rate) = {TPR * 100:.2f}%, FPR (False Positive Rate) = {FPR * 100:.2f}%")
        else:
            logger.warning("No detection outputs available; skipping detection metrics.")
        return metrics


# If running this file directly, run a simple test stub.
if __name__ == "__main__":
    # For testing, create a dummy test dataset (list of dicts).
    from PIL import Image
    import numpy as np

    # Create 10 dummy samples with random images and labels.
    dummy_samples = []
    for i in range(10):
        # Create a dummy RGB PIL image of size 32x32.
        dummy_img = Image.fromarray(np.uint8(np.random.rand(32, 32, 3) * 255))
        # Set a random label between 0 and 9.
        dummy_label = np.random.randint(0, 10)
        # Simulate meta: mark as dirty if i is even.
        meta = {"trigger_injected": (i % 2 == 0), "noisy_injected": False}
        # Optionally, simulate a detector output.
        # For demonstration, set a numeric detection score: lower than 0.5 for dirty samples.
        detected_dirty = 0.3 if meta["trigger_injected"] else 0.7
        sample_dict = {
            "image": dummy_img,
            "label": dummy_label,
            "meta": meta,
            # Do not include "is_dirty" to force fallback computation.
            "detected_dirty": detected_dirty
        }
        dummy_samples.append(sample_dict)

    # Load configuration.
    cfg = load_config("config.yaml")
    # For evaluation, we assume dataset name is specified in training config; if not, default to "CIFAR-10".
    training_cfg = cfg.get("training", {})
    if "dataset_name" not in training_cfg:
        training_cfg["dataset_name"] = "CIFAR-10"
    # Instantiate a dummy model: a simple linear model for classification.
    # Here we create a dummy model that outputs random predictions (for demonstration purposes).
    class DummyModel(torch.nn.Module):
        def __init__(self, num_classes: int = 10):
            super().__init__()
            self.fc = torch.nn.Linear(32 * 32 * 3, num_classes)
        def forward(self, x):
            # Flatten the image tensor.
            x = x.view(x.size(0), -1)
            return self.fc(x)
    dummy_model = DummyModel(num_classes=10)
    # Set the model to evaluation mode.
    dummy_model.eval()

    # Instantiate Evaluation object.
    evaluator = Evaluation(dummy_model, dummy_samples, config=cfg)
    # Run evaluation.
    results = evaluator.evaluate()
    print("Evaluation Results:", results)
