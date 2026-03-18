"""
evaluation.py

This module implements the Evaluation class for evaluating both the classifier and the detection performance.
It computes the following metrics:
  - Classifier Evaluation:
      • Accuracy (ACC) on clean samples.
      • Attack Success Rate (ASR) on attacked (dirty) samples.
  - Detection Evaluation:
      • True Positive Rate (TPR) and False Positive Rate (FPR) for determining how well the detector
        identifies dirty samples.

The Evaluation class requires as input:
  • A trained classifier model (e.g., ResNet-18).
  • A test dataset (loaded via dataset_loader.py) where each sample is expected to be a tuple:
      (image, true_label, dirty_flag)
    The dirty_flag indicates if the sample is attacked/dirty.
  • An optional detector instance (conforming to the VDCDetector interface with the method detect_sample(image, label)),
    which is used for detection evaluation.
    
Configuration parameters (such as the dirty flag attribute name and target label for ASR)
are read from config.yaml.
If absent, default values are used:
  - dirty_flag_name defaults to "dirty"
  - target_label defaults to 0
  - Batch size for evaluation is set to 1 (sample-wise evaluation)

At the end, the evaluate() method returns a dictionary with keys:
  "Accuracy", "Attack_Success_Rate", "True_Positive_Rate", "False_Positive_Rate"
which can be further logged and analyzed.
"""

import os
import logging
from typing import Any, Dict, List, Tuple, Optional

import torch
from torch.utils.data import DataLoader, Dataset

# Import configuration loader from utils
from utils import load_config

# Set up logger.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


class Evaluation:
    """
    Evaluation class for computing classifier and detector metrics.

    Attributes:
        model (torch.nn.Module): Trained classifier model.
        test_data (Dataset): Test dataset with each sample expected as (image, true_label, dirty_flag).
        detector (Optional[Any]): Detector instance implementing detect_sample(image, label).
        config (Dict[str, Any]): Configuration dictionary from config.yaml.
        dirty_flag_name (str): Name of the flag in each sample that indicates an attacked (dirty) sample.
        target_label (int): The target label for attacked samples for computing ASR.
        batch_size (int): Batch size for evaluation DataLoader (default = 1).
        device (torch.device): Device on which the model is located.
    """

    def __init__(self,
                 model: torch.nn.Module,
                 test_data: Dataset,
                 detector: Optional[Any] = None,
                 config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initializes the Evaluation instance with the classifier model, test dataset,
        optional detector, and configuration parameters.

        Args:
            model (torch.nn.Module): Trained classifier model.
            test_data (Dataset): Test dataset; each sample should be a tuple (image, true_label, dirty_flag).
            detector (Optional[Any]): Detector instance (must implement detect_sample(image, label)).
            config (Optional[Dict[str, Any]]): Configuration dictionary. If None, loads from "config.yaml".
        """
        self.model = model
        self.test_data = test_data
        self.detector = detector

        # Load configuration from config.yaml if not provided.
        if config is None:
            config_path = os.path.join(os.getcwd(), "config.yaml")
            try:
                self.config = load_config(config_path)
                logger.info(f"Configuration loaded from {config_path}.")
            except Exception as e:
                logger.error(f"Failed to load configuration: {e}. Using empty configuration.")
                self.config = {}
        else:
            self.config = config

        # Retrieve evaluation-specific configuration.
        self.dirty_flag_name: str = self.config.get("evaluation", {}).get("dirty_flag", "dirty")
        self.target_label: int = self.config.get("evaluation", {}).get("target_label", 0)
        self.batch_size: int = 1  # Evaluation is performed sample-wise.

        logger.info(f"Evaluation initialized with dirty_flag_name='{self.dirty_flag_name}' and target_label={self.target_label}.")

        # Set device based on model parameters.
        self.device: torch.device = next(self.model.parameters()).device if next(self.model.parameters(), None) is not None else torch.device("cpu")
        self.model.eval()

        # Verify detector interface, if provided.
        if self.detector is not None:
            if not hasattr(self.detector, "detect_sample"):
                raise TypeError("The provided detector does not implement a 'detect_sample(image, label)' method.")
            else:
                logger.info("Detector instance provided and conforms to expected interface.")

    def evaluate_classifier(self) -> Dict[str, float]:
        """
        Evaluates the classifier on the test dataset by computing:
          - Accuracy (ACC) on clean (non-attacked) samples.
          - Attack Success Rate (ASR): fraction of attacked samples predicted as the target label.

        Returns:
            Dict[str, float]: Dictionary containing "Accuracy" and "Attack_Success_Rate".
        """
        # Create DataLoader from test_data with batch_size=1.
        data_loader: DataLoader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)

        total_clean: int = 0
        correct_clean: int = 0
        total_attacked: int = 0
        attacked_as_target: int = 0

        with torch.no_grad():
            for sample in data_loader:
                # Expect sample as tuple: (image, true_label, dirty_flag)
                try:
                    if len(sample) == 3:
                        images, labels, dirty_flags = sample
                    elif len(sample) == 2:
                        images, labels = sample
                        # Log warning if dirty flag missing; assume clean.
                        dirty_flags = [False] * images.size(0)
                        logger.warning("Sample missing dirty flag; assuming sample is clean.")
                    else:
                        logger.error("Test sample format invalid. Expected tuple of (image, label, dirty_flag).")
                        continue
                except Exception as e:
                    logger.exception(f"Error unpacking sample: {e}")
                    continue

                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass.
                outputs = self.model(images)
                _, preds = torch.max(outputs, dim=1)

                # Process batch elements (batch_size is 1 for evaluation).
                for i in range(images.size(0)):
                    dirty_flag = dirty_flags[i]
                    true_label: int = labels[i].item()
                    prediction: int = preds[i].item()

                    if dirty_flag:
                        total_attacked += 1
                        if prediction == self.target_label:
                            attacked_as_target += 1
                    else:
                        total_clean += 1
                        if prediction == true_label:
                            correct_clean += 1

        accuracy: float = (correct_clean / total_clean) if total_clean > 0 else 0.0
        asr: float = (attacked_as_target / total_attacked) if total_attacked > 0 else 0.0

        logger.info(f"Classifier Evaluation: Accuracy (clean): {accuracy:.4f}, Attack Success Rate: {asr:.4f}")

        return {"Accuracy": accuracy, "Attack_Success_Rate": asr}

    def evaluate_detection(self) -> Dict[str, float]:
        """
        Evaluates the detector performance on the test dataset.

        For each sample in the test dataset, the detector's decision via detect_sample(image, label)
        is compared with the ground-truth dirty flag. Metrics computed:
          - True Positive Rate (TPR) = TP / (TP + FN)
          - False Positive Rate (FPR) = FP / (FP + TN)

        Returns:
            Dict[str, float]: Dictionary with "True_Positive_Rate" and "False_Positive_Rate".

        Raises:
            ValueError: If no detector instance is provided.
        """
        if self.detector is None:
            raise ValueError("Detector instance is not provided; cannot perform detection evaluation.")

        data_loader: DataLoader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)

        TP, FN, FP, TN = 0, 0, 0, 0

        with torch.no_grad():
            for sample in data_loader:
                # Expected sample tuple: (image, label, dirty_flag)
                try:
                    if len(sample) == 3:
                        image, label, dirty_flags = sample
                    elif len(sample) == 2:
                        image, label = sample
                        dirty_flags = [False] * image.size(0)
                        logger.warning("Sample missing dirty flag during detection evaluation; assuming sample is clean.")
                    else:
                        logger.error("Test sample format invalid during detection evaluation.")
                        continue
                except Exception as e:
                    logger.exception(f"Error unpacking sample during detection evaluation: {e}")
                    continue

                # Since batch_size=1, extract the single sample.
                img = image[0]
                lbl = label[0]
                is_attacked: bool = dirty_flags[0]

                try:
                    # Detector expects image and label (as string typically) 
                    detection_decision: bool = self.detector.detect_sample(img, str(lbl.item()))
                except Exception as e:
                    logger.exception(f"Error in detector.detect_sample: {e}; defaulting decision to clean (False).")
                    detection_decision = False

                if is_attacked:
                    if detection_decision:
                        TP += 1
                    else:
                        FN += 1
                else:
                    if detection_decision:
                        FP += 1
                    else:
                        TN += 1

        TPR: float = (TP / (TP + FN)) if (TP + FN) > 0 else 0.0
        FPR: float = (FP / (FP + TN)) if (FP + TN) > 0 else 0.0

        logger.info(f"Detection Evaluation: TPR: {TPR:.4f}, FPR: {FPR:.4f}; (TP={TP}, FN={FN}, FP={FP}, TN={TN})")

        return {"True_Positive_Rate": TPR, "False_Positive_Rate": FPR}

    def evaluate(self) -> Dict[str, float]:
        """
        Performs the combined evaluation by computing both classifier and detection metrics.
        Returns a dictionary with keys:
            "Accuracy", "Attack_Success_Rate", "True_Positive_Rate", and "False_Positive_Rate".

        Returns:
            Dict[str, float]: Aggregated evaluation metrics.
        """
        results: Dict[str, float] = {}

        # Evaluate classifier performance.
        classifier_metrics = self.evaluate_classifier()
        results.update(classifier_metrics)

        # Evaluate detection performance if a detector is provided.
        if self.detector is not None:
            detection_metrics = self.evaluate_detection()
            results.update(detection_metrics)
        else:
            logger.warning("No detector instance provided; skipping detection evaluation.")

        return results


if __name__ == "__main__":
    # Demonstration of using the Evaluation class.
    import sys
    import numpy as np
    from PIL import Image
    from torch.utils.data import Dataset

    class DummyTestDataset(Dataset):
        """
        Dummy test dataset returning samples as tuples:
            (image, label, dirty_flag)
        Simulates a CIFAR-10 style dataset with 10 classes.
        """
        def __init__(self, num_samples: int, num_classes: int) -> None:
            self.num_samples: int = num_samples
            self.num_classes: int = num_classes
            self.samples: List[Tuple[Image.Image, int, bool]] = []
            for _ in range(num_samples):
                # Create a random 32x32 image.
                img = Image.new("RGB", (32, 32),
                                color=(np.random.randint(0, 256),
                                       np.random.randint(0, 256),
                                       np.random.randint(0, 256)))
                label = np.random.randint(0, num_classes)
                # 30% chance the sample is attacked (dirty).
                dirty = np.random.rand() < 0.3
                self.samples.append((img, label, dirty))

        def __len__(self) -> int:
            return self.num_samples

        def __getitem__(self, index: int) -> Tuple[Any, int, bool]:
            return self.samples[index]

    # Define a dummy classifier model (a simple linear classifier) for demonstration.
    import torch.nn as nn
    class DummyClassifier(nn.Module):
        def __init__(self, num_classes: int):
            super(DummyClassifier, self).__init__()
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(32 * 32 * 3, num_classes)
        def forward(self, x):
            x = self.flatten(x)
            return self.fc(x)
        
    dummy_num_classes: int = 10
    dummy_model = DummyClassifier(num_classes=dummy_num_classes)
    dummy_model.eval()

    # Define a dummy detector that, for demonstration, flags a sample as dirty when the image mean
    # is below a threshold.
    class DummyDetector:
        def detect_sample(self, image: Any, label: str) -> bool:
            import numpy as np
            img_array = np.array(image)
            # For demo: if mean pixel intensity is less than 128, consider it dirty.
            return img_array.mean() < 128

    dummy_detector = DummyDetector()

    # Create a dummy test dataset.
    dummy_test_dataset = DummyTestDataset(num_samples=100, num_classes=dummy_num_classes)

    # Instantiate the Evaluation class.
    evaluation_instance = Evaluation(
        model=dummy_model,
        test_data=dummy_test_dataset,
        detector=dummy_detector
    )

    # Perform evaluation.
    metrics = evaluation_instance.evaluate()
    print("Evaluation Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
