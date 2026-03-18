#!/usr/bin/env python3
"""
main.py

Entry point for the VDC (Versatile Data Cleanser) experiment pipeline.
It orchestrates the overall process:
  1. Loads configuration from "config.yaml" and sets random seeds.
  2. Loads and preprocesses datasets (CIFAR-10, ImageNet-100, ImageNet-Dog) using DatasetLoader.
  3. Injects dirty samples (poisoned samples and noisy labels) into the training set.
  4. Initializes the detection pipeline (VQG, VQA, VAE) and runs dirty sample detection.
  5. Purifies the training dataset by discarding detected dirty samples.
  6. Retrains a ResNet-18 classifier on the purified dataset.
  7. Evaluates the trained classifier on the clean test set (and optionally using backdoor test data).
  
Usage:
    python main.py --dataset CIFAR-10
    (Allowed choices: "CIFAR-10", "ImageNet-100", "ImageNet-Dog")
    
All configuration parameters (e.g., epochs, batch sizes, thresholds) are read from "config.yaml".
"""

import argparse
import os
import logging
from typing import Any, Dict, List, Set

import torch
import torchvision.models as models

# Import project modules
from utils import load_config, get_logger, set_random_seed
from dataset_loader import DatasetLoader, DirtyDataset
from vqg import VQG
from vqa import VQA
from vae import VAE
from detection import VDCDetector
from trainer import Trainer
from evaluation import Evaluation

def get_label_string(sample: Dict[str, Any], dataset_name: str) -> str:
    """
    Converts the numeric label in the sample dictionary to a string label.
    For CIFAR-10, uses a fixed mapping.
    For ImageNet-100 and ImageNet-Dog, uses the "class_name" field if available.
    
    Args:
        sample (Dict[str, Any]): Dictionary representing one dataset sample.
        dataset_name (str): Name of the dataset.
    
    Returns:
        str: The string label.
    """
    if dataset_name == "CIFAR-10":
        cifar10_classes = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ]
        label_index = sample.get("label", 0)
        # Ensure valid index
        if isinstance(label_index, int) and 0 <= label_index < len(cifar10_classes):
            return cifar10_classes[label_index]
        return str(label_index)
    else:
        # For ImageNet-100 or ImageNet-Dog, use "class_name" if present.
        if "class_name" in sample:
            return sample["class_name"]
        # Fallback to string conversion
        return str(sample.get("label", ""))

def main() -> None:
    # ------------------------------
    # 1. Parse Command Line Arguments and Load Config
    # ------------------------------
    parser = argparse.ArgumentParser(description="VDC Experiment Runner")
    parser.add_argument("--dataset", type=str, default="CIFAR-10",
                        choices=["CIFAR-10", "ImageNet-100", "ImageNet-Dog"],
                        help="Dataset to run experiments on. Default: CIFAR-10")
    args = parser.parse_args()
    dataset_name: str = args.dataset

    # Load configuration from config.yaml
    config: Dict[str, Any] = load_config("config.yaml")
    
    # Set random seed for reproducibility (default seed 42 if not specified)
    seed: int = config.get("seed", 42)
    set_random_seed(seed)
    
    # Set up the logger for main
    logger = get_logger("Main")
    logger.info(f"Starting experiment on dataset: {dataset_name}")
    
    # ------------------------------
    # 2. Load and Prepare the Dataset
    # ------------------------------
    dataset_loader: DatasetLoader = DatasetLoader(config)
    data_dict: Dict[str, Dict[str, Any]] = dataset_loader.load_data()
    logger.info("Datasets loaded successfully.")
    
    # Inject dirty samples (poisoned samples + noisy labels) into the training set.
    data_dict = dataset_loader.inject_dirty_samples(data_dict)
    logger.info("Dirty sample injection complete.")
    
    # Extract training and test datasets for the chosen dataset.
    if dataset_name not in data_dict:
        logger.error(f"Dataset {dataset_name} not found in loaded data.")
        return
    train_dataset_dirty: DirtyDataset = data_dict[dataset_name]["train"]
    test_dataset: Any = data_dict[dataset_name]["test"]
    logger.info(f"Training samples (dirty): {len(train_dataset_dirty)}")
    
    # ------------------------------
    # 3. Initialize VDC Detection Pipeline
    # ------------------------------
    # Instantiate VQG module. Use dataset_name and config.
    vqg: VQG = VQG(dataset_name=dataset_name, config=config)
    
    # Instantiate VQA module using instruct_blip model configuration.
    instruct_blip_config: Dict[str, Any] = config.get("api", {}).get("instruct_blip", {"model": "Instruct-BLIP"})
    vqa: VQA = VQA(model_config=instruct_blip_config)
    
    # Instantiate VAE module with evaluation configuration.
    vae: VAE = VAE(eval_config=config)
    
    # Detection threshold from config
    detection_threshold: float = float(config.get("detection", {}).get("threshold", 0.5))
    vdc_detector: VDCDetector = VDCDetector(vqg=vqg, vqa=vqa, vae=vae, threshold=detection_threshold)
    
    logger.info("VDC detection pipeline initialized.")
    
    # ------------------------------
    # 4. Perform Dirty Sample Detection and Purification
    # ------------------------------
    purified_samples: List[Dict[str, Any]] = []
    total_samples: int = len(train_dataset_dirty)
    dirty_count: int = 0
    
    logger.info("Starting dirty sample detection on training data...")
    for idx in range(total_samples):
        sample: Dict[str, Any] = train_dataset_dirty[idx]
        image: Any = sample.get("image")
        # Obtain a string representation of the label
        label_str: str = get_label_string(sample, dataset_name)
        # Use VDCDetector to check if sample is dirty.
        is_dirty: bool = vdc_detector.detect_sample(image=image, label=label_str)
        # Record the detection decision in the sample dictionary.
        sample["detected_dirty"] = is_dirty
        if not is_dirty:
            purified_samples.append(sample)
        else:
            dirty_count += 1
        logger.debug(f"Sample {idx+1}/{total_samples}: Label '{label_str}', Dirty: {is_dirty}")
    
    purified_count: int = len(purified_samples)
    logger.info(f"Dirty sample detection complete: {dirty_count} samples flagged dirty, {purified_count} samples purified out of {total_samples} total.")
    
    # Create purified training dataset
    purified_dataset: DirtyDataset = DirtyDataset(purified_samples)
    
    # ------------------------------
    # 5. Retrain the Classifier on the Purified Dataset
    # ------------------------------
    # Determine number of classes from purified samples.
    unique_labels: Set[Any] = set()
    for sample in purified_samples:
        unique_labels.add(sample.get("label"))
    if len(unique_labels) == 0:
        # Fallback defaults
        if dataset_name == "CIFAR-10":
            num_classes: int = 10
        elif dataset_name == "ImageNet-100":
            num_classes = 100
        elif dataset_name == "ImageNet-Dog":
            num_classes = 10
        else:
            num_classes = 10
    else:
        num_classes = len(unique_labels)
    logger.info(f"Number of classes determined: {num_classes}")
    
    # Instantiate ResNet-18 classifier with the appropriate number of classes.
    classifier: torch.nn.Module = models.resnet18(num_classes=num_classes)
    logger.info("ResNet-18 classifier instantiated.")
    
    # Initialize Trainer with purified dataset and classifier.
    trainer: Trainer = Trainer(model=classifier, train_dataset=purified_dataset,
                               hyperparams=config, dataset_name=dataset_name)
    logger.info("Starting training on purified dataset...")
    trained_model, training_history = trainer.train()
    logger.info("Training completed.")
    
    # ------------------------------
    # 6. Evaluate the Trained Classifier
    # ------------------------------
    # Prepare test_data in dictionary form expected by Evaluation class.
    test_data: Dict[str, Any] = {
        "test": test_dataset,
        "dataset_name": dataset_name
    }
    evaluator: Evaluation = Evaluation(model=trained_model, test_data=test_data)
    evaluation_metrics: Dict[str, float] = evaluator.evaluate()
    logger.info("Evaluation completed.")
    
    # ------------------------------
    # 7. Final Reporting
    # ------------------------------
    logger.info("=== Final Experiment Results ===")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Training samples (dirty): {total_samples}")
    logger.info(f"Dirty samples detected and removed: {dirty_count}")
    logger.info(f"Purified training samples: {purified_count}")
    logger.info(f"Training History: {training_history}")
    logger.info(f"Evaluation Metrics: {evaluation_metrics}")
    
    # Also print the evaluation metrics to stdout.
    print("=== Final Experiment Results ===")
    print(f"Dataset: {dataset_name}")
    print(f"Training samples (dirty): {total_samples}")
    print(f"Dirty samples detected and removed: {dirty_count}")
    print(f"Purified training samples: {purified_count}")
    print(f"Training History: {training_history}")
    print(f"Evaluation Metrics: {evaluation_metrics}")

if __name__ == "__main__":
    main()
