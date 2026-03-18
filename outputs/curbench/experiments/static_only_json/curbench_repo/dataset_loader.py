"""
dataset_loader.py

This module implements the DatasetLoader class which handles loading and preprocessing of datasets 
across computer vision (CV), natural language processing (NLP), and graph domains. It provides a unified 
interface to download raw datasets, perform train/validation/test splits, apply label noise ("noise-p" setting), 
and create imbalanced versions ("imbalance-r" setting) for CV tasks. The configuration parameters (such as 
random seeds, training splits, and preprocessing parameters) are passed via a configuration dictionary 
loaded from "config.yaml" using the Config module.
 
Dependencies:
    - torch, torchvision for CV datasets and splitting.
    - numpy for numerical operations.
    - random for random sampling.
    - datasets (from Hugging Face) for NLP datasets (GLUE).
    - torch_geometric and ogb for graph datasets.
    
All random operations are seeded using the seed provided in the configuration (reproducibility.seeds).
"""

import os
import re
import copy
import logging
import random
from math import floor
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Subset, random_split

# For CV: torchvision is required.
import torchvision
from torchvision import transforms

# For NLP: Try to import Hugging Face datasets.
try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

# For graph datasets:
# ogb might be required for ogbg-molhiv; if not installed, an exception will be raised when used.
try:
    from ogb.graphproppred import PygGraphPropPredDataset
except ImportError:
    PygGraphPropPredDataset = None

from torch_geometric.datasets import TUDataset


class DatasetLoader:
    """
    DatasetLoader loads and preprocesses datasets according to the experiment configuration.
    It supports CV, NLP, and graph domains, and applies noise injection and imbalance creation
    when specified in the dataset identifier string.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initializes the DatasetLoader with the given configuration.

        Args:
            config (Dict[str, Any]): Configuration dictionary loaded from config.yaml.
        """
        self.config: Dict[str, Any] = config
        reproducibility_config: Dict[str, Any] = config.get("reproducibility", {})
        seeds: list = reproducibility_config.get("seeds", [42])
        self.random_seed: int = int(seeds[0])  # use the first seed as default for all random ops
        # Seed Python's random and numpy for reproducibility.
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.info("DatasetLoader initialized with seed: %s", self.random_seed)

    def load_data(self, dataset_identifier: str) -> Dict[str, Any]:
        """
        Unified entry point to load and preprocess a dataset based on the identifier.
        It parses the dataset identifier to determine the domain, applies domain-specific loading,
        and then calls preprocessing functions such as noise injection and imbalance creation.

        Args:
            dataset_identifier (str): A string identifier for the dataset, e.g., "cifar10",
                                      "cifar100-noise-0.4", "tinyimagenet-imbalance-50", etc.

        Returns:
            Dict[str, Any]: A dictionary with keys "train", "val", "test", and "num_classes".
        """
        dataset_id: str = dataset_identifier.lower()
        domain: str = self._infer_domain(dataset_id)

        self.logger.info("Loading dataset '%s' for domain '%s'.", dataset_identifier, domain)

        if domain == "cv":
            data: Dict[str, Any] = self._load_cv_dataset(dataset_id)
        elif domain == "nlp":
            data = self._load_nlp_dataset(dataset_id)
        elif domain == "graph":
            data = self._load_graph_dataset(dataset_id)
        else:
            raise ValueError(f"Unrecognized domain for dataset identifier: {dataset_identifier}")

        # Apply noise injection if specified.
        noise_prob: Optional[float] = self._extract_noise_prob(dataset_id)
        if noise_prob is not None and noise_prob > 0.0:
            self.logger.info("Applying noise injection with probability %.2f.", noise_prob)
            data["train"] = self.apply_noise(data["train"], noise_prob)

        # Apply imbalance for CV datasets if specified.
        imbalance_factor: Optional[float] = self._extract_imbalance_factor(dataset_id)
        if imbalance_factor is not None and imbalance_factor > 1 and domain == "cv":
            self.logger.info("Applying imbalance with factor %.2f.", imbalance_factor)
            data["train"] = self.apply_imbalance(data["train"], imbalance_factor)

        return data

    def _infer_domain(self, dataset_id: str) -> str:
        """
        Infers the domain (cv, nlp, or graph) based on the dataset identifier.

        Args:
            dataset_id (str): The dataset identifier string.

        Returns:
            str: Domain string: "cv", "nlp", or "graph".

        Raises:
            ValueError: If the dataset identifier does not map to any recognized domain.
        """
        if "cifar" in dataset_id or "tinyimagenet" in dataset_id:
            return "cv"
        elif dataset_id in ["rte", "mrpc", "sts-b", "cola", "sst-2", "qnli", "qqp", "mnli"]:
            return "nlp"
        elif "ogbg" in dataset_id or dataset_id in ["mutag", "proteins", "nci1"]:
            return "graph"
        else:
            raise ValueError(f"Dataset identifier '{dataset_id}' is not recognized for domain inference.")

    def _extract_noise_prob(self, dataset_id: str) -> Optional[float]:
        """
        Extracts the noise probability from the dataset identifier if present.

        Args:
            dataset_id (str): The dataset identifier string.

        Returns:
            Optional[float]: The noise probability (e.g., 0.4) if specified, else None.
        """
        match = re.search(r"noise[-_]?(\d+(\.\d+)?)", dataset_id)
        if match:
            return float(match.group(1))
        return None

    def _extract_imbalance_factor(self, dataset_id: str) -> Optional[float]:
        """
        Extracts the imbalance factor from the dataset identifier if present.

        Args:
            dataset_id (str): The dataset identifier string.

        Returns:
            Optional[float]: The imbalance factor (e.g., 50) if specified, else None.
        """
        match = re.search(r"imbalance[-_]?(\d+(\.\d+)?)", dataset_id)
        if match:
            return float(match.group(1))
        return None

    def _load_cv_dataset(self, dataset_id: str) -> Dict[str, Any]:
        """
        Loads a computer vision (CV) dataset based on the dataset identifier.
        For CIFAR-10 and CIFAR-100, uses torchvision datasets; for Tiny-ImageNet, uses ImageFolder.
        Splits the original training set into training and validation sets (ratio 9:1).

        Args:
            dataset_id (str): The dataset identifier string.

        Returns:
            Dict[str, Any]: A dictionary with keys "train", "val", "test", and "num_classes".
        """
        transform = transforms.Compose([transforms.ToTensor()])
        dataset_id_lower: str = dataset_id.lower()
        train_dataset: Any
        test_dataset: Any

        if "cifar10" in dataset_id_lower:
            train_dataset = torchvision.datasets.CIFAR10(
                root="./data", train=True, download=True, transform=transform
            )
            test_dataset = torchvision.datasets.CIFAR10(
                root="./data", train=False, download=True, transform=transform
            )
        elif "cifar100" in dataset_id_lower:
            train_dataset = torchvision.datasets.CIFAR100(
                root="./data", train=True, download=True, transform=transform
            )
            test_dataset = torchvision.datasets.CIFAR100(
                root="./data", train=False, download=True, transform=transform
            )
        elif "tinyimagenet" in dataset_id_lower:
            # Assume Tiny-ImageNet data is organized under "./data/tiny-imagenet-200"
            from torchvision.datasets import ImageFolder

            train_folder: str = os.path.join("./data", "tiny-imagenet-200", "train")
            val_folder: str = os.path.join("./data", "tiny-imagenet-200", "val")
            if not os.path.exists(train_folder) or not os.path.exists(val_folder):
                raise FileNotFoundError("Tiny-ImageNet dataset folders not found. Please ensure dataset is located at './data/tiny-imagenet-200'.")
            full_train_dataset = ImageFolder(root=train_folder, transform=transform)
            train_dataset, val_dataset = self._cv_train_val_split(full_train_dataset, split_ratio=0.9)
            # For Tiny-ImageNet, use the provided validation folder as the test set.
            test_dataset = ImageFolder(root=val_folder, transform=transform)
            return {
                "train": train_dataset,
                "val": val_dataset,
                "test": test_dataset,
                "num_classes": len(full_train_dataset.classes)
            }
        else:
            raise ValueError(f"Unrecognized CV dataset identifier: {dataset_id}")

        train_split, val_split = self._cv_train_val_split(train_dataset, split_ratio=0.9)
        num_classes: int = len(train_dataset.classes)
        return {"train": train_split, "val": val_split, "test": test_dataset, "num_classes": num_classes}

    def _cv_train_val_split(self, dataset: Any, split_ratio: float = 0.9) -> Tuple[Any, Any]:
        """
        Splits a CV dataset into training and validation subsets using the given ratio.

        Args:
            dataset (Any): The complete dataset to split.
            split_ratio (float): Fraction of data to use for training (default 0.9).

        Returns:
            Tuple[Any, Any]: A tuple containing (train_subset, val_subset).
        """
        dataset_length: int = len(dataset)
        train_length: int = int(split_ratio * dataset_length)
        val_length: int = dataset_length - train_length
        generator = torch.Generator().manual_seed(self.random_seed)
        train_subset, val_subset = random_split(dataset, [train_length, val_length], generator=generator)
        return train_subset, val_subset

    def _load_nlp_dataset(self, dataset_id: str) -> Dict[str, Any]:
        """
        Loads an NLP dataset from the GLUE benchmark using Hugging Face's datasets library.
        Uses the training and validation splits provided by the dataset, with the validation set
        also serving as the test set (as per paper instructions).

        Args:
            dataset_id (str): The dataset identifier string (e.g., "rte", "mrpc", "sts-b", etc.)

        Returns:
            Dict[str, Any]: A dictionary with keys "train", "val", "test", and "num_classes".
        """
        if load_dataset is None:
            raise ImportError("The 'datasets' library is required for NLP data loading. Please install it via 'pip install datasets'.")
        glue_task: str = dataset_id.lower()
        dataset = load_dataset("glue", glue_task)
        train_dataset = dataset["train"]
        val_dataset = dataset["validation"]
        test_dataset = dataset["validation"]  # Using validation as test as specified
        # Determine number of classes; default to 2 if not inferable.
        num_classes: int = 2
        if "label" in train_dataset.features:
            labels = train_dataset["label"]
            unique_labels = set(labels)
            num_classes = len(unique_labels)
        return {"train": train_dataset, "val": val_dataset, "test": test_dataset, "num_classes": num_classes}

    def _load_graph_dataset(self, dataset_id: str) -> Dict[str, Any]:
        """
        Loads a graph dataset. For 'ogbg-molhiv', uses the OGB's PyG dataset with provided splits.
        For other graph datasets (e.g., "mutag", "proteins", "nci1"), uses the TUDataset and creates a 
        random train/validation/test split (8:1:1).

        Args:
            dataset_id (str): The dataset identifier string.

        Returns:
            Dict[str, Any]: A dictionary with keys "train", "val", "test", and "num_classes".
        """
        dataset_id_lower: str = dataset_id.lower()
        if "ogbg-molhiv" in dataset_id_lower:
            if PygGraphPropPredDataset is None:
                raise ImportError("The 'ogb' package is required for loading ogbg-molhiv. Please install it.")
            dataset = PygGraphPropPredDataset(name="ogbg-molhiv", root="./data")
            splits = dataset.get_idx_split()
            train_dataset = dataset[splits["train"]]
            val_dataset = dataset[splits["valid"]]
            test_dataset = dataset[splits["test"]]
            num_classes: int = 2
            return {"train": train_dataset, "val": val_dataset, "test": test_dataset, "num_classes": num_classes}
        else:
            # Assume dataset_id corresponds to a TUDataset.
            dataset = TUDataset(root="./data", name=dataset_id.upper())
            total: int = len(dataset)
            train_count: int = int(0.8 * total)
            val_count: int = int(0.1 * total)
            test_count: int = total - train_count - val_count
            generator = torch.Generator().manual_seed(self.random_seed)
            train_dataset, val_dataset, test_dataset = random_split(dataset, [train_count, val_count, test_count], generator=generator)
            num_classes: int = dataset.num_classes
            return {"train": train_dataset, "val": val_dataset, "test": test_dataset, "num_classes": num_classes}

    def apply_noise(self, dataset: Any, noise_prob: float) -> Any:
        """
        Applies label noise to the training dataset by flipping each label with probability 'noise_prob'.
        For CV datasets (torchvision), it operates on the 'targets' attribute. For NLP datasets from Hugging Face,
        a map function is applied.

        Args:
            dataset (Any): The training dataset to modify.
            noise_prob (float): The probability with which a label is changed.

        Returns:
            Any: The modified dataset with noise applied.

        Raises:
            ValueError: If the dataset does not support noise injection.
        """
        # For torchvision datasets that have 'targets' attribute.
        if hasattr(dataset, "targets"):
            noisy_dataset = copy.deepcopy(dataset)
            if hasattr(noisy_dataset, "classes"):
                num_classes: int = len(noisy_dataset.classes)
            else:
                num_classes = int(max(noisy_dataset.targets)) + 1
            new_targets = []
            for orig_label in noisy_dataset.targets:
                if np.random.rand() < noise_prob:
                    candidate_labels = list(range(num_classes))
                    candidate_labels.remove(orig_label)
                    new_label = int(np.random.choice(candidate_labels))
                    new_targets.append(new_label)
                else:
                    new_targets.append(orig_label)
            noisy_dataset.targets = new_targets
            return noisy_dataset
        # For Hugging Face datasets (NLP), use the map function.
        elif hasattr(dataset, "feature") or ("label" in dataset.column_names):
            def noise_fn(example):
                orig_label = example["label"]
                if np.random.rand() < noise_prob:
                    # Determine all possible labels from the dataset's feature info if available, else infer from current label.
                    labels = list(set(dataset["label"]))
                    candidate_labels = [lbl for lbl in labels if lbl != orig_label]
                    if candidate_labels:
                        example["label"] = int(np.random.choice(candidate_labels))
                return example
            return dataset.map(noise_fn)
        else:
            raise ValueError("Dataset does not support noise injection: missing 'targets' attribute or recognizable label field.")

    def apply_imbalance(self, dataset: Any, imbalance_factor: float) -> Any:
        """
        Creates an imbalanced version of a CV training dataset by sub-sampling each class based on an exponential decay.
        The decay factor d is computed as: d = (1/imbalance_factor)^(1/(C-1)), where C is the number of classes.
        For each class c, the new count is floor(n0 * d^c), where n0 is the original count for the first class.

        Args:
            dataset (Any): The training dataset, assumed to have a 'targets' attribute.
            imbalance_factor (float): The imbalance factor r (must be >= 1). r = 1 means no imbalance.

        Returns:
            Any: A torch.utils.data.Subset representing the imbalanced training dataset.

        Raises:
            ValueError: If the dataset does not support imbalance injection.
        """
        if not hasattr(dataset, "targets"):
            raise ValueError("Dataset does not support imbalance injection: missing 'targets' attribute.")

        original_targets: np.ndarray = np.array(dataset.targets)
        unique_classes = np.unique(original_targets)
        num_classes: int = len(unique_classes)

        # Assume original dataset is balanced; use count of first class as n0.
        counts_per_class = {cls: np.sum(original_targets == cls) for cls in unique_classes}
        n0: int = int(counts_per_class[unique_classes[0]])
        # Compute decay factor: d = (1 / imbalance_factor)^(1/(C-1))
        d: float = (1.0 / imbalance_factor) ** (1.0 / (num_classes - 1))
        selected_indices = []
        # Ensure classes are processed in sorted order.
        for c_index, cls in enumerate(sorted(unique_classes)):
            desired_count: int = int(floor(n0 * (d ** c_index)))
            class_indices = np.where(original_targets == cls)[0]
            if desired_count > len(class_indices):
                desired_count = len(class_indices)
            sampled_indices = np.random.choice(class_indices, size=desired_count, replace=False)
            selected_indices.extend(sampled_indices.tolist())
        imbalanced_dataset = Subset(dataset, selected_indices)
        return imbalanced_dataset
