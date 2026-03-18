"""
dataset_loader.py

This module implements the DatasetLoader class which loads the CIFAR-10, ImageNet-100,
and ImageNet-Dog datasets and injects dirty samples (backdoor poisoning and noisy labels)
according to the experimental configuration provided in config.yaml.

The DatasetLoader uses torchvision for data loading, performs reproducible sub-sampling
per class (using fixed seeds), and applies dirty sample injection using utility functions
(e.g., for BadNets backdoor trigger injection and noisy label flipping). Dirty injections
are mutually exclusive by default unless hybrid mode is explicitly enabled.
"""

import os
import random
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
import torchvision
from torchvision import transforms

# Import utility functions for trigger injection and label noise.
# It is assumed that utils.py provides at least the following functions:
#   - inject_badnets_trigger(image: Any) -> Any
#   - flip_label_noise(true_label: int, num_classes: int) -> int
try:
    import utils
except ImportError:
    logging.error("utils module not found. Please ensure utils.py exists with required functions.")
    # To allow the module to run for testing, define dummy functions.
    class DummyUtils:
        @staticmethod
        def inject_badnets_trigger(image: Any) -> Any:
            # Dummy trigger injection: simply return the image unchanged.
            return image

        @staticmethod
        def flip_label_noise(true_label: int, num_classes: int) -> int:
            available = list(range(num_classes))
            if true_label in available:
                available.remove(true_label)
            return random.choice(available) if available else true_label

    utils = DummyUtils()

# Set up logger
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


class ListDataset(Dataset):
    """A simple Dataset wrapper that holds a list of (image, label) tuples."""
    def __init__(self, samples: List[Tuple[Any, int]]):
        """
        Args:
            samples (List[Tuple[Any, int]]): A list where each element is a tuple (image, label)
        """
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        return self.samples[index]


class DatasetLoader:
    """
    DatasetLoader loads the CIFAR-10, ImageNet-100, and ImageNet-Dog datasets,
    performs fixed sub-sampling to ensure reproducibility and consistent splits,
    and provides a method to inject dirty samples (poisoned samples and noisy labels)
    into the training data.
    
    Methods:
        __init__(config: dict, poisoning_level: Optional[Dict[str, int]] = None, hybrid_dirty: bool = False)
        load_data() -> Dict[str, Dict[str, Dataset]]
        inject_dirty_samples(data: Dict[str, Dict[str, Dataset]]) -> Dict[str, Dict[str, Dataset]]
    """
    def __init__(
        self,
        config: Dict[str, Any],
        poisoning_level: Optional[Dict[str, int]] = None,
        hybrid_dirty: bool = False
    ) -> None:
        """
        Args:
            config (dict): Configuration dictionary loaded from config.yaml.
            poisoning_level (Optional[Dict[str, int]]): An optional dictionary specifying the poisoning sample 
                count per class for each dataset (e.g., {"CIFAR-10": 50, "ImageNet-100": 5, "ImageNet-Dog": 80}).
                If not provided, default values are chosen from config.
            hybrid_dirty (bool): Flag to specify if hybrid dirty samples should be generated (i.e. both poisoning and noisy label injection applied to a sample).
                                 Default is False (mutually exclusive injection).
        """
        self.config = config
        self.hybrid_dirty = hybrid_dirty
        # Set a fixed seed for reproducibility
        self.seed: int = 42
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Process poisoning levels from configuration.
        # The config field dirty_sample_generation.poisoning is expected for each dataset.
        # For datasets with a list of sample counts (e.g., CIFAR-10 and ImageNet-100), we choose the first value by default,
        # unless provided in the optional poisoning_level argument.
        self.poisoning_levels: Dict[str, int] = {}
        poisoning_config: Dict[str, Any] = self.config.get("dirty_sample_generation", {}).get("poisoning", {})
        for dataset_name, level_value in poisoning_config.items():
            if isinstance(level_value, list):
                # Use provided poisoning level if available; otherwise select the first value in the list.
                selected_level = poisoning_level.get(dataset_name) if (poisoning_level and dataset_name in poisoning_level) else level_value[0]
                if selected_level not in level_value:
                    raise ValueError(
                        f"Invalid poisoning level for {dataset_name}. Allowed values: {level_value}, got: {selected_level}"
                    )
                self.poisoning_levels[dataset_name] = selected_level
            elif isinstance(level_value, int):
                self.poisoning_levels[dataset_name] = level_value
            else:
                raise ValueError(f"Invalid type for poisoning configuration of {dataset_name}: {type(level_value)}")

        # Retrieve dirty injection noisy label ratio.
        self.noisy_ratio: float = self.config.get("dirty_sample_generation", {}).get("noisy_ratio", 0.4)
        logging.info(f"Initialized DatasetLoader with poisoning_levels: {self.poisoning_levels} and noisy_ratio: {self.noisy_ratio}")

    def load_data(self) -> Dict[str, Dict[str, Dataset]]:
        """
        Loads CIFAR-10, ImageNet-100, and ImageNet-Dog datasets. For ImageNet datasets, a fixed random
        sampling per class is used (with fixed seeds) to select a specified number of training and test samples:
            - ImageNet-100: 500 training and 100 testing samples per class.
            - ImageNet-Dog: 800 training and 200 testing samples per class.
        
        Returns:
            Dict[str, Dict[str, Dataset]]: A dictionary with keys "CIFAR-10", "ImageNet-100", "ImageNet-Dog".
            Each value is another dictionary with keys "train" and "test" containing the respective dataset objects.
        """
        data: Dict[str, Dict[str, Dataset]] = {}

        # ---------------------------
        # Load CIFAR-10 dataset
        # ---------------------------
        cifar_train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                 std=(0.2023, 0.1994, 0.2010))
        ])
        cifar_test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                 std=(0.2023, 0.1994, 0.2010))
        ])
        cifar_train = torchvision.datasets.CIFAR10(
            root="./data/cifar10", train=True, download=True, transform=cifar_train_transform
        )
        cifar_test = torchvision.datasets.CIFAR10(
            root="./data/cifar10", train=False, download=True, transform=cifar_test_transform
        )
        data["CIFAR-10"] = {"train": cifar_train, "test": cifar_test}
        logging.info("Loaded CIFAR-10 dataset.")

        # ---------------------------
        # Load ImageNet-100 dataset
        # ---------------------------
        # Assumed directory structure:
        #   Train: "./data/imagenet_100/train"
        #   Test:  "./data/imagenet_100/test"
        imagenet100_train_dir = "./data/imagenet_100/train"
        imagenet100_test_dir = "./data/imagenet_100/test"
        imagenet_transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ])
        imagenet_transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ])
        if os.path.isdir(imagenet100_train_dir) and os.path.isdir(imagenet100_test_dir):
            imagenet100_train_full = torchvision.datasets.ImageFolder(
                root=imagenet100_train_dir, transform=imagenet_transform_train
            )
            imagenet100_test_full = torchvision.datasets.ImageFolder(
                root=imagenet100_test_dir, transform=imagenet_transform_test
            )
            # Randomly select 100 classes if necessary.
            all_classes = list(range(len(imagenet100_train_full.classes)))
            if len(all_classes) > 100:
                rng = random.Random(self.seed)
                selected_classes = rng.sample(all_classes, 100)
            else:
                selected_classes = all_classes
            # For training: sample 500 samples per selected class.
            imagenet100_train = self._create_subset(
                dataset=imagenet100_train_full,
                samples_per_class=500,
                selected_classes=selected_classes,
                seed=self.seed
            )
            # For test: sample 100 samples per selected class.
            imagenet100_test = self._create_subset(
                dataset=imagenet100_test_full,
                samples_per_class=100,
                selected_classes=selected_classes,
                seed=self.seed
            )
            data["ImageNet-100"] = {"train": imagenet100_train, "test": imagenet100_test}
            logging.info("Loaded ImageNet-100 dataset with {} classes.".format(len(selected_classes)))
        else:
            logging.warning("ImageNet-100 directories not found. Skipping ImageNet-100 dataset.")

        # ---------------------------
        # Load ImageNet-Dog dataset
        # ---------------------------
        # Assumed directory structure:
        #   Train: "./data/imagenet_dog/train"
        #   Test:  "./data/imagenet_dog/test"
        imagenetdog_train_dir = "./data/imagenet_dog/train"
        imagenetdog_test_dir = "./data/imagenet_dog/test"
        if os.path.isdir(imagenetdog_train_dir) and os.path.isdir(imagenetdog_test_dir):
            imagenetdog_train_full = torchvision.datasets.ImageFolder(
                root=imagenetdog_train_dir, transform=imagenet_transform_train
            )
            imagenetdog_test_full = torchvision.datasets.ImageFolder(
                root=imagenetdog_test_dir, transform=imagenet_transform_test
            )
            # For ImageNet-Dog, assume the dataset already contains only dog classes.
            imagenetdog_train = self._create_subset(
                dataset=imagenetdog_train_full,
                samples_per_class=800,
                selected_classes=None,
                seed=self.seed
            )
            imagenetdog_test = self._create_subset(
                dataset=imagenetdog_test_full,
                samples_per_class=200,
                selected_classes=None,
                seed=self.seed
            )
            data["ImageNet-Dog"] = {"train": imagenetdog_train, "test": imagenetdog_test}
            logging.info("Loaded ImageNet-Dog dataset.")
        else:
            logging.warning("ImageNet-Dog directories not found. Skipping ImageNet-Dog dataset.")

        return data

    def _create_subset(
        self,
        dataset: Dataset,
        samples_per_class: int,
        selected_classes: Optional[List[int]] = None,
        seed: int = 42
    ) -> Subset:
        """
        Creates a subset of the given dataset that contains exactly `samples_per_class` samples for each class.
        If selected_classes is provided, only those classes are considered.
        
        Args:
            dataset (Dataset): The original dataset.
            samples_per_class (int): Number of samples to select per class.
            selected_classes (Optional[List[int]]): List of class indices to include.
            seed (int): Random seed for reproducibility.
        
        Returns:
            Subset: A torch.utils.data.Subset containing the selected indices.
        """
        class_to_indices: Dict[int, List[int]] = {}
        # Loop over all indices in dataset
        for idx in range(len(dataset)):
            sample = dataset[idx]  # Expected to be a tuple (image, label)
            if not isinstance(sample, (list, tuple)) or len(sample) < 2:
                continue
            _, label = sample
            if selected_classes is not None and label not in selected_classes:
                continue
            if label not in class_to_indices:
                class_to_indices[label] = []
            class_to_indices[label].append(idx)

        selected_indices: List[int] = []
        rng = random.Random(seed)
        for label, indices in class_to_indices.items():
            if len(indices) >= samples_per_class:
                sampled = rng.sample(indices, samples_per_class)
            else:
                logging.warning(f"Class {label}: Only {len(indices)} samples available, required {samples_per_class}.")
                sampled = indices
            selected_indices.extend(sampled)
        return Subset(dataset, selected_indices)

    def inject_dirty_samples(self, data: Dict[str, Dict[str, Dataset]]) -> Dict[str, Dict[str, Dataset]]:
        """
        Injects dirty samples into the training data of each dataset.
        The injection applies two mechanisms:
          1. Poisoned sample injection (e.g., using a BadNets trigger) is applied to a pre-determined number of samples per class.
          2. Noisy label injection is applied to the remaining samples with probability = noisy_ratio.
        By default, these injections are mutually exclusive. Hybrid mode (poison + noisy) can be enabled via the hybrid_dirty flag.
        
        Args:
            data (Dict[str, Dict[str, Dataset]]): The dataset dictionary as returned by load_data(),
                                                   with keys "train" and "test" for each dataset.
        
        Returns:
            Dict[str, Dict[str, Dataset]]: The modified dataset dictionary (only the training data is altered).
        """
        modified_data = data.copy()
        # Process each dataset in the dictionary
        for dataset_name, splits in data.items():
            if "train" not in splits:
                continue  # Skip if no training split found.
            train_subset = splits["train"]
            # For Subset objects, the original dataset is stored in train_subset.dataset and indices in train_subset.indices.
            original_dataset = train_subset.dataset
            subset_indices: List[int] = train_subset.indices

            # Group samples by label.
            label_groups: Dict[int, List[Tuple[int, Any, int]]] = {}
            for idx in subset_indices:
                sample = original_dataset[idx]  # (image, label)
                if not isinstance(sample, (list, tuple)) or len(sample) < 2:
                    continue
                image, label = sample
                if label not in label_groups:
                    label_groups[label] = []
                label_groups[label].append((idx, image, label))
            
            # Determine number of classes for noisy label injection.
            # Try to use original_dataset.classes if available.
            num_classes: int = len(getattr(original_dataset, "classes", []))
            if num_classes == 0:
                # Default to a safe number based on known dataset.
                if dataset_name == "CIFAR-10":
                    num_classes = 10
                elif dataset_name == "ImageNet-100":
                    num_classes = 100
                elif dataset_name == "ImageNet-Dog":
                    num_classes = 10
                else:
                    num_classes = 10

            logging.info(f"Injecting dirty samples into {dataset_name}: num_classes = {num_classes}")

            # Prepare new sample list after dirty injection.
            new_samples: List[Tuple[Any, int]] = []
            # For each class label, perform injection.
            for label, samples in label_groups.items():
                # Create a random generator instance with a seed offset by label for reproducibility.
                rng = random.Random(self.seed + label)
                poisoning_count: int = self.poisoning_levels.get(dataset_name, 0)
                if poisoning_count > len(samples):
                    logging.warning(f"For dataset {dataset_name}, label {label}: requested poisoning_count {poisoning_count} exceeds available samples ({len(samples)}). Using all available samples for poisoning.")
                    poisoning_samples = samples
                else:
                    poisoning_samples = rng.sample(samples, poisoning_count)

                poisoning_indices_set = set([item[0] for item in poisoning_samples])

                # Process each sample in this group.
                for (idx, image, true_label) in samples:
                    new_image = image
                    new_label = true_label
                    if idx in poisoning_indices_set:
                        # Apply backdoor trigger injection (e.g., BadNets trigger) to the image.
                        new_image = self._apply_poisoning(image)
                        # For mutually exclusive injection, do not apply noisy label change.
                    else:
                        # For non-poisoned samples, decide whether to inject noisy label.
                        if self.hybrid_dirty:
                            # In hybrid mode, apply both poisoning trigger and noise.
                            new_image = self._apply_poisoning(image)
                            new_label = self._apply_noisy_label(true_label, num_classes)
                        else:
                            # Default: apply noisy label injection probabilistically.
                            if rng.random() < self.noisy_ratio:
                                new_label = self._apply_noisy_label(true_label, num_classes)
                    new_samples.append((new_image, new_label))
            # Replace training split with the modified list wrapped in a ListDataset.
            modified_data[dataset_name]["train"] = ListDataset(new_samples)
            logging.info(f"Injected dirty samples into dataset {dataset_name}: total training samples modified = {len(new_samples)}")
        return modified_data

    def _apply_poisoning(self, image: Any) -> Any:
        """
        Applies a backdoor trigger injection to the provided image using the BadNets method.
        
        Args:
            image (Any): The input image.
        
        Returns:
            Any: The modified image with the injected trigger.
        """
        try:
            modified_image = utils.inject_badnets_trigger(image)
            return modified_image
        except Exception as e:
            logging.error(f"Error in applying poisoning trigger: {e}")
            return image

    def _apply_noisy_label(self, true_label: int, num_classes: int) -> int:
        """
        Flips the label to a noisy label that is different from the true label.
        
        Args:
            true_label (int): The original label.
            num_classes (int): Total number of classes.
        
        Returns:
            int: A noisy label chosen from the other labels.
        """
        try:
            noisy_label = utils.flip_label_noise(true_label, num_classes)
            return noisy_label
        except Exception as e:
            logging.error(f"Error in applying noisy label injection: {e}")
            # Fallback: perform random flip manually.
            available_labels = list(range(num_classes))
            if true_label in available_labels:
                available_labels.remove(true_label)
            return random.choice(available_labels) if available_labels else true_label
