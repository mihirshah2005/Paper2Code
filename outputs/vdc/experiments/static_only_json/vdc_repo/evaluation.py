"""evaluation.py

This module defines the Evaluation class, which evaluates a trained classifier 
(e.g., ResNet-18) on test datasets. It computes key metrics including:
  - Clean Accuracy (ACC) on the clean test set.
  - Attack Success Rate (ASR) on triggered (backdoor) test samples.
  - Detection performance metrics (TPR and FPR) if a detection test set with 
    ground truth dirty flags is provided.

The Evaluation class expects test_data as a dictionary containing at least 
a "test" key (the clean test dataset). Optionally, it can include:
  - "poisoned": a test dataset with pre-triggered images.
  - "detection": a dataset with per sample detection flags ("is_dirty" as ground truth
      and "detected_dirty" as the detector's output).

This file uses configuration from "config.yaml" (via utils.load_config) to set 
the batch size for evaluation and other experimental settings. In addition, 
if a poisoned test set is not provided, the Evaluation class will generate triggered 
versions (using the default "BadNets" trigger) on-the-fly from clean test samples.

Author: [Your Name]
Date: [Date]
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Any, Dict, List

import torchvision.transforms as transforms

# Import helper functions and configuration utilities.
from utils import load_config, get_logger, inject_backdoor_trigger

# ------------------------------------------------------------------------------
# Evaluation Class Definition
# ------------------------------------------------------------------------------

class Evaluation:
    """
    Evaluation class for computing performance metrics of a trained classifier.
    
    Attributes:
        model (Any): Trained classifier (e.g., ResNet-18) for evaluation.
        test_data (Dict[str, Any]): Dictionary containing test datasets.
            Expected key:
              - "test": Clean test dataset.
            Optional keys:
              - "poisoned": Poisoned/triggered test dataset.
              - "detection": Dataset with detection flags (each sample should include
                             "is_dirty" (ground truth) and "detected_dirty" (detector output)).
              - "dataset_name": Name of the dataset (e.g., "CIFAR-10")
        config (Dict[str, Any]): Experiment configuration loaded from config.yaml.
        logger (logging.Logger): Logger for output messages.
        device (torch.device): Device for running evaluation.
        batch_size (int): Batch size used during evaluation.
        dataset_name (str): Name of the dataset for evaluation.
        clean_test_loader (DataLoader): DataLoader for clean test data.
        poisoned_loader (DataLoader): DataLoader for poisoned test data (if provided).
        detection_loader (DataLoader): DataLoader for detection evaluation (if provided).
        target_label (int): The target label for backdoor attacks (default is 0).
    """
    
    def __init__(self, model: Any, test_data: Dict[str, Any]) -> None:
        """
        Initializes the Evaluation instance.
        
        Args:
            model (Any): The trained classifier (e.g., ResNet-18).
            test_data (Dict[str, Any]): Dictionary containing test datasets.
                Must contain the key "test" for the clean test set.
                Optionally, include "poisoned" for triggered images and 
                "detection" for detection evaluation.
        """
        self.model: Any = model
        self.test_data: Dict[str, Any] = test_data
        self.config: Dict[str, Any] = load_config("config.yaml")
        self.logger = get_logger("Evaluation")
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        # Determine dataset name; default to "CIFAR-10" if not provided.
        self.dataset_name: str = self.test_data.get("dataset_name", "CIFAR-10")
        
        # Retrieve batch size from config for the given dataset.
        training_config: Dict[str, Any] = self.config.get("training", {})
        batch_size_config: Dict[str, Any] = training_config.get("batch_size", {})
        self.batch_size: int = int(batch_size_config.get(self.dataset_name, 128))
        
        # Create DataLoader for clean test data.
        self.clean_test_loader: DataLoader = DataLoader(
            self.test_data["test"],
            batch_size=self.batch_size,
            shuffle=False
        )
        
        # Create DataLoader for detection evaluation if provided.
        if "detection" in self.test_data:
            self.detection_loader: DataLoader = DataLoader(
                self.test_data["detection"],
                batch_size=self.batch_size,
                shuffle=False
            )
        else:
            self.detection_loader = None
        
        # Create DataLoader for poisoned test data if provided.
        if "poisoned" in self.test_data:
            self.poisoned_loader: DataLoader = DataLoader(
                self.test_data["poisoned"],
                batch_size=self.batch_size,
                shuffle=False
            )
        else:
            self.poisoned_loader = None
        
        # The target label for backdoor attacks is assumed to be 0.
        self.target_label: int = 0

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluates the classifier on the provided test datasets.
        
        Computes:
          - ACC_clean: Clean Accuracy (percentage) on the clean test set.
          - ASR_poisoned: Attack Success Rate (percentage) on triggered test samples.
          - TPR_detection and FPR_detection if a detection dataset is provided.
        
        Returns:
            Dict[str, float]: A summary dictionary containing evaluation metrics.
        """
        results: Dict[str, float] = {}
        
        # ----------------------------
        # Evaluate Clean Accuracy (ACC)
        # ----------------------------
        total_clean: int = 0
        correct_clean: int = 0
        
        self.model.eval()
        with torch.no_grad():
            for batch in self.clean_test_loader:
                # Batch may be a tuple (inputs, labels) or a dict with keys.
                if isinstance(batch, dict):
                    images = batch.get("image")
                    labels = batch.get("label")
                elif isinstance(batch, (list, tuple)):
                    images, labels = batch[0], batch[1]
                else:
                    continue
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                predictions = outputs.argmax(dim=1)
                correct_clean += (predictions == labels).sum().item()
                total_clean += labels.size(0)
        
        acc_clean: float = (correct_clean / total_clean * 100.0) if total_clean > 0 else 0.0
        results["ACC_clean"] = acc_clean
        self.logger.info(f"Clean Accuracy (ACC): {acc_clean:.2f}%")
        
        # ----------------------------
        # Evaluate Attack Success Rate (ASR)
        # ----------------------------
        total_poisoned: int = 0
        successful_poison: int = 0
        
        if self.poisoned_loader is not None:
            # Use provided poisoned/test dataset.
            with torch.no_grad():
                for batch in self.poisoned_loader:
                    if isinstance(batch, dict):
                        images = batch.get("image")
                        _ = batch.get("label")  # not needed for ASR evaluation
                    elif isinstance(batch, (list, tuple)):
                        images, _ = batch[0], batch[1]
                    else:
                        continue
                    images = images.to(self.device)
                    outputs = self.model(images)
                    predictions = outputs.argmax(dim=1)
                    successful_poison += (predictions == self.target_label).sum().item()
                    total_poisoned += images.size(0)
        else:
            # If no separate poisoned test set is provided, generate triggered images on-the-fly.
            triggered_images_list: List[torch.Tensor] = []
            # We need to apply the same backdoor trigger as in training (default: "BadNets").
            # Use a standard transformation for the given dataset.
            with torch.no_grad():
                # Use a temporary transform based on dataset.
                if self.dataset_name == "CIFAR-10":
                    transform_fn = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                             std=(0.2470, 0.2435, 0.2616))
                    ])
                else:
                    transform_fn = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
                    ])
                
                for batch in self.clean_test_loader:
                    if isinstance(batch, dict):
                        images = batch.get("image")
                    elif isinstance(batch, (list, tuple)):
                        images = batch[0]
                    else:
                        continue
                    # Process each image individually.
                    for img in images:
                        # If image is a tensor, convert to PIL image.
                        if not hasattr(img, "convert"):
                            from torchvision.transforms import ToPILImage
                            pil_img = ToPILImage()(img.cpu())
                        else:
                            pil_img = img
                        # Apply the default backdoor trigger.
                        triggered_pil = inject_backdoor_trigger(pil_img, trigger_type="BadNets", dataset=self.dataset_name)
                        tensor_img = transform_fn(triggered_pil)
                        triggered_images_list.append(tensor_img)
                
                if triggered_images_list:
                    triggered_tensor = torch.stack(triggered_images_list).to(self.device)
                    outputs = self.model(triggered_tensor)
                    predictions = outputs.argmax(dim=1)
                    successful_poison = (predictions == self.target_label).sum().item()
                    total_poisoned = triggered_tensor.size(0)
        
        asr_poisoned: float = (successful_poison / total_poisoned * 100.0) if total_poisoned > 0 else 0.0
        results["ASR_poisoned"] = asr_poisoned
        self.logger.info(f"Attack Success Rate (ASR): {asr_poisoned:.2f}% (Target label: {self.target_label})")
        
        # ----------------------------
        # Evaluate Detection Metrics (TPR and FPR), if available.
        # ----------------------------
        if self.detection_loader is not None:
            true_positives: int = 0
            false_negatives: int = 0
            false_positives: int = 0
            true_negatives: int = 0
            
            # Expect each sample to have keys "is_dirty" (ground truth) and "detected_dirty" (detector's flag).
            for batch in self.detection_loader:
                if not isinstance(batch, dict):
                    continue
                detected_flags = batch.get("detected_dirty")
                ground_truth_flags = batch.get("is_dirty")
                # Convert tensor values to lists if necessary.
                if isinstance(detected_flags, torch.Tensor):
                    detected_list = detected_flags.cpu().tolist()
                else:
                    detected_list = detected_flags
                if isinstance(ground_truth_flags, torch.Tensor):
                    ground_truth_list = ground_truth_flags.cpu().tolist()
                else:
                    ground_truth_list = ground_truth_flags
                for detected, truth in zip(detected_list, ground_truth_list):
                    if truth and detected:
                        true_positives += 1
                    elif truth and not detected:
                        false_negatives += 1
                    elif not truth and detected:
                        false_positives += 1
                    elif not truth and not detected:
                        true_negatives += 1
            
            tpr_detection: float = (true_positives / (true_positives + false_negatives) * 100.0) if (true_positives + false_negatives) > 0 else 0.0
            fpr_detection: float = (false_positives / (false_positives + true_negatives) * 100.0) if (false_positives + true_negatives) > 0 else 0.0
            results["TPR_detection"] = tpr_detection
            results["FPR_detection"] = fpr_detection
            self.logger.info(f"Detection Metrics - TPR: {tpr_detection:.2f}%, FPR: {fpr_detection:.2f}%")
        else:
            self.logger.info("Detection evaluation skipped (no detection dataset provided).")
        
        return results

# ------------------------------------------------------------------------------
# Standalone Testing
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    import torchvision.datasets as datasets
    from torchvision import models
    import torchvision.transforms as transforms
    from torch.utils.data import Dataset
    
    # Load configuration.
    config = load_config("config.yaml")
    
    # Use CIFAR-10 test set as an example.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                             std=(0.2470, 0.2435, 0.2616))
    ])
    cifar10_test = datasets.CIFAR10(root="./data/cifar10", train=False, download=True, transform=transform)
    
    # Prepare test_data dictionary.
    test_data: Dict[str, Any] = {"test": cifar10_test, "dataset_name": "CIFAR-10"}
    
    # Optionally, simulate a detection dataset by adding synthetic flags.
    detection_samples: List[Dict[str, Any]] = []
    for idx in range(len(cifar10_test)):
        image, label = cifar10_test[idx]
        # For demonstration, mark roughly 10% of samples as dirty.
        is_dirty: bool = (idx % 10 == 0)
        # Simulate detector outputs with some noise.
        detected_dirty: bool = is_dirty if idx % 3 != 0 else not is_dirty
        detection_samples.append({
            "image": image,
            "label": label,
            "is_dirty": is_dirty,
            "detected_dirty": detected_dirty
        })
    
    class SimpleDetectionDataset(Dataset):
        def __init__(self, samples: List[Dict[str, Any]]) -> None:
            self.samples = samples
        def __len__(self) -> int:
            return len(self.samples)
        def __getitem__(self, idx: int) -> Dict[str, Any]:
            return self.samples[idx]
    
    detection_dataset = SimpleDetectionDataset(detection_samples)
    test_data["detection"] = detection_dataset
    
    # For demonstration, we do not supply a separate poisoned dataset.
    
    # Instantiate an example classifier (ResNet-18) for CIFAR-10.
    model = models.resnet18(num_classes=10)
    # (In practice, load your trained model here.)
    
    evaluator = Evaluation(model=model, test_data=test_data)
    metrics = evaluator.evaluate()
    print("Evaluation Metrics:", metrics)
"""