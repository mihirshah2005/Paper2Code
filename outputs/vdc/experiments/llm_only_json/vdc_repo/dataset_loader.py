"""
dataset_loader.py

This module implements the DatasetLoader class which is responsible for loading the datasets
(CIFAR-10, ImageNet-100, ImageNet-Dog), applying image transformations, and injecting dirty samples
(e.g., backdoor poisoning and noisy labels) into the training set. The class behavior is driven by
the provided configuration (from config.yaml) and ensures reproducibility via fixed seeding.

The overall design:
    - __init__(config: dict, dataset_name: str) sets the seed, config, and initializes two augmentation
      pipelines: a standard pipeline for training/testing and a trigger-preserving pipeline for samples
      injected with a backdoor trigger.
    - load_data() loads the dataset in a deterministic and reproducible manner, building a unified list
      of sample dictionaries with keys "image", "label", and "meta" (metadata flags).
    - inject_dirty_samples(data: dict) modifies the training data by injecting backdoor triggers (using a
      simple BadNets approach) and then applying noisy label modifications in a mutually exclusive manner
      per class.
      
This file depends on the following packages:
    numpy==1.21.0, torch==1.9.0, torchvision==0.10.0, transformers==4.15.0,
    openai==0.27.0, requests==2.26.0

Usage:
    from dataset_loader import DatasetLoader
    config = ...  # a configuration dict loaded from config.yaml
    loader = DatasetLoader(config, dataset_name="CIFAR-10")
    data = loader.load_data()
    purified_data = loader.inject_dirty_samples(data)
"""

import os
import random
import logging
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
from torchvision import datasets, transforms
from PIL import Image, ImageDraw

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


def inject_poison_badnets(image: Image.Image, dataset_name: str) -> Image.Image:
    """
    Injects a simple BadNets trigger into the input image.
    For CIFAR-10, a 3x3 white square is drawn in the lower-right corner.
    For other datasets (e.g. ImageNet-100, ImageNet-Dog), a 21x21 white square is used.
    
    Args:
        image: A PIL Image.
        dataset_name: Name of the dataset to determine trigger size.
    
    Returns:
        Modified PIL Image with trigger injected.
    """
    # Ensure image is in RGB mode
    if image.mode != "RGB":
        image = image.convert("RGB")
    width, height = image.size
    draw = ImageDraw.Draw(image)
    if dataset_name == "CIFAR-10":
        trigger_size = 3
    else:
        trigger_size = 21
    # Coordinates for bottom-right corner trigger
    x0 = width - trigger_size
    y0 = height - trigger_size
    x1 = width
    y1 = height
    # Draw a white square trigger
    draw.rectangle([x0, y0, x1, y1], fill=(255, 255, 255))
    return image


def inject_noisy_label(original_label: int, num_classes: int, noise_type: str = "symmetric") -> int:
    """
    Generates a noisy label based on the original label.
    
    For symmetric noise, a random label (other than the original) is chosen.
    For asymmetric noise, the new label is set as (original_label + 1) mod num_classes.
    
    Args:
        original_label: The original label as an integer.
        num_classes: Total number of classes.
        noise_type: "symmetric" or "asymmetric" noise injection.
    
    Returns:
        An integer representing the noisy label.
    """
    if noise_type == "symmetric":
        possible_labels = list(range(num_classes))
        if original_label in possible_labels:
            possible_labels.remove(original_label)
        return random.choice(possible_labels)
    elif noise_type == "asymmetric":
        return (original_label + 1) % num_classes
    else:
        # Default to symmetric if not recognized.
        possible_labels = list(range(num_classes))
        possible_labels.remove(original_label)
        return random.choice(possible_labels)


class DatasetLoader:
    """
    DatasetLoader handles the loading of CIFAR-10, ImageNet-100, and ImageNet-Dog datasets.
    It also injects dirty samples (poisoning and noisy labels) in a reproducible fashion.
    
    Attributes:
        config (Dict[str, Any]): Configuration dictionary.
        dataset_name (str): Name of the dataset ("CIFAR-10", "ImageNet-100", or "ImageNet-Dog").
        seed (int): Random seed used for deterministic behavior.
        transform_train (transforms.Compose): Standard training pipeline.
        transform_test (transforms.Compose): Standard testing pipeline.
        trigger_transform (transforms.Compose): Trigger-preserving augmentation pipeline.
    """
    def __init__(self, config: Dict[str, Any], dataset_name: str = "CIFAR-10") -> None:
        self.config = config
        self.dataset_name = dataset_name
        self.seed: int = self.config.get("seed", 42)
        # Set global seeds for reproducibility
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        
        # Initialize data transformation pipelines based on dataset
        if self.dataset_name == "CIFAR-10":
            # Standard training transformation: data augmentation + normalization
            self.transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))
            ])
            # Standard testing transformation: only normalization
            self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))
            ])
            # Trigger-preserving transformation: no spatial augmentation
            self.trigger_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))
            ])
            self.num_classes = 10
        elif self.dataset_name in ["ImageNet-100", "ImageNet-Dog"]:
            # For ImageNet-based datasets, assume images are of size 224x224.
            # Standard training transformation with data augmentation:
            self.transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            # Standard testing transformation:
            self.transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            # Trigger-preserving transformation: minimal geometric augmentation.
            self.trigger_transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            # Set number of classes based on dataset type
            if self.dataset_name == "ImageNet-100":
                self.num_classes = 100
            else:  # ImageNet-Dog
                self.num_classes = 10
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        logging.info(f"DatasetLoader initialized for {self.dataset_name} with seed {self.seed}")

    def load_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Loads the dataset and returns a dictionary with train and test data.
        Each data sample is represented as a dictionary with keys:
            - "image": PIL Image (raw image, with no transform applied)
            - "label": int
            - "meta": Dict for metadata (e.g., flags for modifications)
        
        Returns:
            Dict with keys "train_data" and "test_data", each a list of sample dicts.
        """
        data: Dict[str, List[Dict[str, Any]]] = {"train_data": [], "test_data": []}
        
        if self.dataset_name == "CIFAR-10":
            # Load CIFAR-10 without any transform; transforms are applied later
            train_dataset = datasets.CIFAR10(root="./data/cifar10", train=True, download=True, transform=None)
            test_dataset = datasets.CIFAR10(root="./data/cifar10", train=False, download=True, transform=None)
            # Process train dataset
            for img, label in train_dataset:
                sample = {
                    "image": img,  # PIL Image
                    "label": label,
                    "meta": {"trigger_injected": False, "noisy_injected": False}
                }
                data["train_data"].append(sample)
            # Process test dataset
            for img, label in test_dataset:
                sample = {
                    "image": img,
                    "label": label,
                    "meta": {"trigger_injected": False, "noisy_injected": False}
                }
                data["test_data"].append(sample)
            logging.info(f"CIFAR-10 loaded: {len(data['train_data'])} train samples, {len(data['test_data'])} test samples.")
        
        elif self.dataset_name in ["ImageNet-100", "ImageNet-Dog"]:
            # For ImageNet-based datasets, we assume the images are stored in a folder structure
            # that is compatible with torchvision.datasets.ImageFolder.
            # The root directory is assumed to be "./data/imagenet".
            root_dir = f"./data/{self.dataset_name.lower()}"
            if not os.path.exists(root_dir):
                raise FileNotFoundError(f"Dataset path {root_dir} does not exist.")
            full_dataset = datasets.ImageFolder(root=root_dir, transform=None)
            logging.info(f"Full ImageFolder dataset loaded from {root_dir} with {len(full_dataset)} samples.")
            # Determine selected classes.
            all_classes: List[str] = full_dataset.classes  # e.g., list of folder names
            selected_classes: List[str] = []
            # Check if configuration provides a subset list; otherwise select deterministically.
            subset_key = f"{self.dataset_name}_subset_classes"
            if subset_key in self.config:
                selected_classes = self.config[subset_key]
                logging.info(f"Using user-provided subset classes for {self.dataset_name}: {selected_classes}")
            else:
                if self.dataset_name == "ImageNet-100":
                    # Deterministically select 100 classes using the fixed seed.
                    selected_classes = sorted(random.sample(all_classes, 100))
                    logging.info(f"Deterministically selected 100 classes for ImageNet-100: {selected_classes}")
                elif self.dataset_name == "ImageNet-Dog":
                    # Use a predefined list of 10 dog classes if not provided.
                    selected_classes = [
                        "Chihuahua", "Japanese_spaniel", "Maltese_dog", "Pekinese",
                        "Shih-Tzu", "Blenheim_spaniel", "papillon", "toy_terrier",
                        "Rhodesian_ridgeback", "Afghan_hound"
                    ]
                    logging.info(f"Using default 10 dog classes for ImageNet-Dog: {selected_classes}")
                else:
                    raise ValueError(f"Unsupported ImageNet dataset type: {self.dataset_name}")
            # Map selected class names to their indices.
            selected_class_indices = [all_classes.index(cls) for cls in selected_classes if cls in all_classes]
            if len(selected_class_indices) != len(selected_classes):
                logging.warning("Some selected classes were not found in the dataset classes.")
            # Group samples by class index.
            groups: Dict[int, List[Tuple[str, int]]] = {}
            for sample in full_dataset.samples:
                path, label = sample
                if label in selected_class_indices:
                    groups.setdefault(label, []).append(sample)
            # Define per-class sample quotas based on dataset type.
            if self.dataset_name == "ImageNet-100":
                train_quota, test_quota = 500, 100
            else:  # ImageNet-Dog
                train_quota, test_quota = 800, 200
            train_data: List[Dict[str, Any]] = []
            test_data: List[Dict[str, Any]] = []
            for label, samples_list in groups.items():
                # Sort samples deterministically by filename
                samples_list = sorted(samples_list, key=lambda x: x[0])
                available = len(samples_list)
                if available < train_quota + test_quota:
                    logging.warning(f"Class index {label} has only {available} samples; "
                                    f"requested {train_quota + test_quota}. Using available samples.")
                # Partition samples deterministically.
                train_samples = samples_list[:min(train_quota, available)]
                test_samples = samples_list[min(train_quota, available): min(train_quota + test_quota, available)]
                for path, lbl in train_samples:
                    try:
                        img = Image.open(path).convert("RGB")
                    except Exception as e:
                        logging.error(f"Error loading image {path}: {e}")
                        continue
                    sample_dict = {
                        "image": img,
                        "label": lbl,
                        "meta": {"trigger_injected": False, "noisy_injected": False}
                    }
                    train_data.append(sample_dict)
                for path, lbl in test_samples:
                    try:
                        img = Image.open(path).convert("RGB")
                    except Exception as e:
                        logging.error(f"Error loading image {path}: {e}")
                        continue
                    sample_dict = {
                        "image": img,
                        "label": lbl,
                        "meta": {"trigger_injected": False, "noisy_injected": False}
                    }
                    test_data.append(sample_dict)
            data["train_data"] = train_data
            data["test_data"] = test_data
            logging.info(f"{self.dataset_name} loaded: {len(train_data)} train samples, {len(test_data)} test samples.")
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
        
        return data

    def inject_dirty_samples(self, data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Injects dirty samples (poisoned samples and noisy labels) into the training data.
        The injection process is performed per class in a mutually exclusive manner:
            1. Poisoning (backdoor triggers) is applied first using a configured number of samples.
            2. Then noisy label modifications are applied to a fraction (noisy_ratio) of the remaining samples.
        
        Modifications:
            - Poisoned samples are injected using a simple BadNets trigger (a white square) and 
              their labels are set to a fixed target (0).
            - Noisy label injection: For symmetric noise, a random label (other than the original) is used.
        
        Args:
            data: Dictionary with "train_data" key containing a list of sample dicts.
            
        Returns:
            The modified data dictionary with injected dirty samples.
        """
        train_data: List[Dict[str, Any]] = data.get("train_data", [])
        # Group sample indices by their original label.
        groups: Dict[int, List[int]] = {}
        for idx, sample in enumerate(train_data):
            lbl = sample["label"]
            groups.setdefault(lbl, []).append(idx)
        
        # Read poisoning parameters from configuration
        poison_config = self.config.get("dirty_sample_generation", {}).get("poisoning", {})
        if self.dataset_name not in poison_config:
            logging.warning(f"No poisoning configuration found for {self.dataset_name}. Skipping poisoning injection.")
            poison_count = 0
        else:
            samples_per_class = poison_config[self.dataset_name].get("samples_per_class")
            # If samples_per_class is a list, use the first element as default.
            if isinstance(samples_per_class, list):
                poison_count = samples_per_class[0]
            else:
                poison_count = samples_per_class

        noisy_ratio: float = self.config.get("dirty_sample_generation", {}).get("noisy_ratio", 0.4)
        
        # Iterate over each class group.
        for lbl, indices in groups.items():
            num_samples_in_class = len(indices)
            # Determine number of samples to poison
            num_poison = min(poison_count, num_samples_in_class)
            if num_poison < poison_count:
                logging.warning(f"Class {lbl}: requested poison count {poison_count} but only {num_samples_in_class} samples available.")
            poison_indices = random.sample(indices, num_poison) if num_poison > 0 else []
            # Mark poisoned samples
            for idx in poison_indices:
                sample = train_data[idx]
                original_image = sample["image"]
                # Inject poison using the BadNets method.
                poisoned_image = inject_poison_badnets(original_image.copy(), self.dataset_name)
                sample["image"] = poisoned_image
                sample["label"] = 0  # Set target label for poisoning as 0.
                sample["meta"]["trigger_injected"] = True
            # Mark samples used for noisy label injection (exclude already poisoned samples)
            remaining_indices = [i for i in indices if i not in poison_indices]
            num_remaining = len(remaining_indices)
            num_noisy = int(noisy_ratio * num_remaining)
            if num_noisy > num_remaining:
                logging.warning(f"Class {lbl}: requested noisy count {num_noisy} exceeds available {num_remaining}. Using {num_remaining} samples.")
                num_noisy = num_remaining
            noisy_indices = random.sample(remaining_indices, num_noisy) if num_noisy > 0 else []
            # For noisy label injection, update the label using symmetric noise injection.
            for idx in noisy_indices:
                sample = train_data[idx]
                orig_label = sample["label"]
                new_label = inject_noisy_label(orig_label, self.num_classes, noise_type="symmetric")
                sample["label"] = new_label
                sample["meta"]["noisy_injected"] = True
        logging.info("Dirty sample injection completed on training data.")
        data["train_data"] = train_data
        return data

    @staticmethod
    def worker_init_fn(worker_id: int, base_seed: int = 42) -> None:
        """
        Initializes each worker with a seed so that data loading with multiple workers is deterministic.
        
        Args:
            worker_id: The worker's id.
            base_seed: Base seed to combine with worker_id.
        """
        worker_seed = base_seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
