"""
dataset_loader.py

This module contains the DatasetLoader class which handles loading and preprocessing
of the CIFAR-10, ImageNet-100, and ImageNet-Dog datasets, as well as simulating dirty
sample injection (noisy labels and backdoor triggers) into the training data.
"""

import os
import random
import numpy as np
import logging
from typing import Dict, Any, List
import torch
from torch.utils.data import Subset
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageDraw

# Set up basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class DatasetLoader:
    """Handles loading datasets, applying image transformations, and injecting dirty samples.

    The dirty sample injection process consists of:
      1. Noisy label injection: With probability η, the label is modified using symmetric noise.
      2. Backdoor (poisoned) injection: For non-target classes, a fixed number of samples per class
         are selected. A trigger is injected into the image and the label is changed to the target (0).
    
    Attributes:
        config (dict): Configuration dictionary parsed from config.yaml.
        noisy_ratio (float): Probability of injecting noisy label per sample.
        poisoning_config (dict): Configuration for number of poisoned samples per class.
        seed (int): Random seed value.
        cifar_transform_train: Transform pipeline for CIFAR-10 training images.
        cifar_transform_test: Transform pipeline for CIFAR-10 test images.
        imagenet_transform_train: Transform pipeline for ImageNet-based training images.
        imagenet_transform_test: Transform pipeline for ImageNet-based test images.
        imagenet100_train_path (str): Root directory for ImageNet-100 training data.
        imagenet100_val_path (str): Root directory for ImageNet-100 validation/test data.
        imagenet_dog_train_path (str): Root directory for ImageNet-Dog training data.
        imagenet_dog_val_path (str): Root directory for ImageNet-Dog validation/test data.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize DatasetLoader with configuration settings.

        Args:
            config: The configuration dictionary (from config.yaml).
        """
        self.config = config

        # Set random seeds for reproducibility.
        self.seed: int = config.get("seed", 42)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Dirty sample generation parameters.
        dirty_config = config.get("dirty_sample_generation", {})
        self.noisy_ratio: float = dirty_config.get("noisy_ratio", 0.4)
        self.poisoning_config: Dict[str, Any] = dirty_config.get("poisoning", {})

        # Setup dataset paths for ImageNet-based datasets.
        dataset_paths: Dict[str, Any] = config.get("dataset_paths", {})
        self.imagenet100_train_path: str = dataset_paths.get("ImageNet-100_train", "./data/imagenet100/train")
        self.imagenet100_val_path: str = dataset_paths.get("ImageNet-100_val", "./data/imagenet100/val")
        self.imagenet_dog_train_path: str = dataset_paths.get("ImageNet-Dog_train", "./data/imagenet_dog/train")
        self.imagenet_dog_val_path: str = dataset_paths.get("ImageNet-Dog_val", "./data/imagenet_dog/val")

        # Setup transforms for CIFAR-10.
        self.cifar_transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616))
        ])
        self.cifar_transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616))
        ])

        # Setup transforms for ImageNet-based datasets (ImageNet-100 and ImageNet-Dog).
        self.imagenet_transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))
        ])
        self.imagenet_transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))
        ])

    def load_data(self) -> Dict[str, Dict[str, Any]]:
        """Load and preprocess the datasets.

        Returns:
            A dictionary with keys 'CIFAR-10', 'ImageNet-100', and 'ImageNet-Dog'.
            Each value is a dictionary with keys 'train' and 'test' corresponding to the dataset splits.
        """
        data_dict: Dict[str, Dict[str, Any]] = {}

        # Load CIFAR-10 dataset.
        logging.info("Loading CIFAR-10 dataset...")
        cifar_train = torchvision.datasets.CIFAR10(
            root="./data/cifar10",
            train=True,
            download=True,
            transform=self.cifar_transform_train
        )
        cifar_test = torchvision.datasets.CIFAR10(
            root="./data/cifar10",
            train=False,
            download=True,
            transform=self.cifar_transform_test
        )
        data_dict["CIFAR-10"] = {"train": cifar_train, "test": cifar_test}

        # Load ImageNet-100 dataset using ImageFolder.
        logging.info("Loading ImageNet-100 dataset...")
        imagenet100_train_full = torchvision.datasets.ImageFolder(
            root=self.imagenet100_train_path,
            transform=self.imagenet_transform_train
        )
        imagenet100_test_full = torchvision.datasets.ImageFolder(
            root=self.imagenet100_val_path,
            transform=self.imagenet_transform_test
        )
        # Randomly select 100 classes.
        all_classes = list(range(len(imagenet100_train_full.classes)))
        selected_classes = random.sample(all_classes, 100)
        logging.info(f"Selected classes for ImageNet-100: {selected_classes}")
        # Filter and subsample: 500 train images per class and 100 test images per class.
        imagenet100_train = self._filter_dataset(
            dataset=imagenet100_train_full,
            selected_classes=selected_classes,
            samples_per_class=500
        )
        imagenet100_test = self._filter_dataset(
            dataset=imagenet100_test_full,
            selected_classes=selected_classes,
            samples_per_class=100
        )
        data_dict["ImageNet-100"] = {"train": imagenet100_train, "test": imagenet100_test}

        # Load ImageNet-Dog dataset using ImageFolder.
        logging.info("Loading ImageNet-Dog dataset...")
        imagenet_dog_train_full = torchvision.datasets.ImageFolder(
            root=self.imagenet_dog_train_path,
            transform=self.imagenet_transform_train
        )
        imagenet_dog_test_full = torchvision.datasets.ImageFolder(
            root=self.imagenet_dog_val_path,
            transform=self.imagenet_transform_test
        )
        # Randomly select 10 dog classes.
        all_dog_classes = list(range(len(imagenet_dog_train_full.classes)))
        selected_dog_classes = random.sample(all_dog_classes, 10)
        logging.info(f"Selected classes for ImageNet-Dog: {selected_dog_classes}")
        # Filter and subsample: 800 train images per class and 200 test images per class.
        imagenet_dog_train = self._filter_dataset(
            dataset=imagenet_dog_train_full,
            selected_classes=selected_dog_classes,
            samples_per_class=800
        )
        imagenet_dog_test = self._filter_dataset(
            dataset=imagenet_dog_test_full,
            selected_classes=selected_dog_classes,
            samples_per_class=200
        )
        data_dict["ImageNet-Dog"] = {"train": imagenet_dog_train, "test": imagenet_dog_test}

        return data_dict

    def inject_dirty_samples(self, data: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Inject noisy labels and backdoor triggers into the training datasets.

        The injection is performed in two steps:
          1. Noisy Label Injection:
             - For each training sample, with probability equal to noisy_ratio, the label is modified.
             - Symmetric noise is applied (randomly choosing a label different from the original).
          2. Backdoor (Poisoned) Injection:
             - For each non-target class (target label is assumed to be 0), a fixed number of samples per class
               (as specified in the configuration) are randomly selected.
             - For each selected sample, a trigger is injected into the image and the label is set to the target (0).

        Args:
            data: Dictionary containing training and test datasets for each dataset.

        Returns:
            Modified data dictionary with dirty injections applied to training datasets.
        """
        for dataset_name, splits in data.items():
            train_dataset = splits["train"]

            # Determine number of classes.
            if hasattr(train_dataset, "classes"):
                num_classes = len(train_dataset.classes)
            elif hasattr(train_dataset, "targets"):
                # Default to 10 classes if target information is not explicit (e.g., CIFAR-10).
                num_classes = 10
            else:
                num_classes = 10

            # --- Noisy Label Injection ---
            logging.info(f"Injecting noisy labels into {dataset_name} training data...")
            if isinstance(train_dataset, torchvision.datasets.CIFAR10):
                # Modify labels in CIFAR-10 using train_dataset.targets.
                for idx in range(len(train_dataset.targets)):
                    if random.random() < self.noisy_ratio:
                        original_label = train_dataset.targets[idx]
                        new_label = random.choice([lbl for lbl in range(num_classes) if lbl != original_label])
                        train_dataset.targets[idx] = new_label
            else:
                # For ImageFolder-based datasets.
                if hasattr(train_dataset, "targets") and isinstance(train_dataset.targets, list):
                    for idx in range(len(train_dataset.targets)):
                        if random.random() < self.noisy_ratio:
                            original_label = train_dataset.targets[idx]
                            new_label = random.choice([lbl for lbl in range(num_classes) if lbl != original_label])
                            train_dataset.targets[idx] = new_label

            # --- Backdoor (Poisoned) Sample Injection ---
            logging.info(f"Injecting backdoor triggers into {dataset_name} training data...")
            poisoning_value = self.poisoning_config.get(dataset_name, None)
            if poisoning_value is None:
                logging.info(f"No poisoning configuration for {dataset_name}; skipping backdoor injection.")
                continue
            # If poisoning_value is a list, use the first element as the default.
            if isinstance(poisoning_value, list):
                poison_samples_per_class = poisoning_value[0]
            else:
                poison_samples_per_class = poisoning_value

            # Create mapping from class labels to indices.
            class_to_indices: Dict[int, List[int]] = {}
            if isinstance(train_dataset, torchvision.datasets.CIFAR10):
                for idx, label in enumerate(train_dataset.targets):
                    class_to_indices.setdefault(label, []).append(idx)
            else:
                for idx, label in enumerate(train_dataset.targets):
                    class_to_indices.setdefault(label, []).append(idx)

            target_label = 0  # The designated target label for backdoor samples.
            for cls in range(num_classes):
                if cls == target_label:
                    continue
                indices = class_to_indices.get(cls, [])
                if not indices:
                    continue
                selected_indices = random.sample(indices, min(poison_samples_per_class, len(indices)))
                for idx in selected_indices:
                    # Inject trigger into the image.
                    if isinstance(train_dataset, torchvision.datasets.CIFAR10):
                        original_image = train_dataset.data[idx]
                        modified_image = self._inject_trigger_cifar(original_image)
                        train_dataset.data[idx] = modified_image
                    else:
                        # For ImageFolder-based datasets, load the image, modify it, and store an override.
                        image_path, _ = train_dataset.samples[idx]
                        try:
                            original_image = Image.open(image_path).convert("RGB")
                        except Exception as e:
                            logging.error(f"Error loading image {image_path}: {e}")
                            continue
                        modified_image = self._inject_trigger_imagenet(original_image)
                        if not hasattr(train_dataset, "dirty_overrides"):
                            train_dataset.dirty_overrides = {}
                        train_dataset.dirty_overrides[idx] = modified_image
                    # Set label to target.
                    if isinstance(train_dataset, torchvision.datasets.CIFAR10):
                        train_dataset.targets[idx] = target_label
                    else:
                        train_dataset.targets[idx] = target_label

            logging.info(f"Completed dirty sample injection for {dataset_name}.")
        return data

    def _filter_dataset(
        self, dataset: Any, selected_classes: List[int], samples_per_class: int
    ) -> Subset:
        """Filter and subsample a dataset to include only selected classes with a fixed number per class.

        Args:
            dataset: The original dataset (e.g., an ImageFolder instance).
            selected_classes: List of class indices to retain.
            samples_per_class: Maximum number of samples to keep per class.

        Returns:
            A torch.utils.data.Subset containing the selected samples.
        """
        indices_to_keep: List[int] = []
        # Determine target labels.
        if hasattr(dataset, "targets") and isinstance(dataset.targets, list):
            targets = dataset.targets
        else:
            targets = [s[1] for s in dataset.samples]
        # Create a mapping from class label to list of indices.
        class_to_indices: Dict[int, List[int]] = {}
        for idx, label in enumerate(targets):
            if label in selected_classes:
                class_to_indices.setdefault(label, []).append(idx)
        # For each selected class, choose a random subset of indices.
        for cls in selected_classes:
            cls_indices = class_to_indices.get(cls, [])
            if not cls_indices:
                continue
            selected = random.sample(cls_indices, min(samples_per_class, len(cls_indices)))
            indices_to_keep.extend(selected)
        return Subset(dataset, indices_to_keep)

    def _inject_trigger_cifar(self, image: np.ndarray) -> np.ndarray:
        """Inject a backdoor trigger into a CIFAR-10 image.

        This function overlays a 3x3 white square in the lower-right corner of the image.

        Args:
            image: A numpy array representing the image (H, W, C).

        Returns:
            The modified image as a numpy array.
        """
        image_copy = image.copy()
        h, w, _ = image_copy.shape
        trigger_size = 3
        image_copy[h - trigger_size : h, w - trigger_size : w, :] = 255
        return image_copy

    def _inject_trigger_imagenet(self, image: Image.Image) -> Image.Image:
        """Inject a backdoor trigger into an ImageNet image.

        This function overlays a 21x21 white square in the lower-right corner of the image.

        Args:
            image: A PIL Image object representing the image.

        Returns:
            The modified PIL Image with the trigger applied.
        """
        image_copy = image.copy()
        draw = ImageDraw.Draw(image_copy)
        width, height = image_copy.size
        trigger_size = 21
        draw.rectangle([width - trigger_size, height - trigger_size, width, height], fill=(255, 255, 255))
        return image_copy
