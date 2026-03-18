"""
dataset_loader.py

This module contains the DatasetLoader class for loading, splitting, and preprocessing
datasets across Computer Vision (CV), Natural Language Processing (NLP), and Graph domains.
It implements robust parsing of dataset identifiers (e.g., "cifar10-noise-0.4", "tinyimagenet-imbalance-50")
and applies noise injection and long-tailed imbalance modifications as required.

The DatasetLoader sets seeds for reproducibility across Python's random,
NumPy, and PyTorch (both CPU and GPU) and configures deterministic behavior.

External dependencies:
    - os, re, math, random, logging, copy, collections
    - numpy
    - torch, torchvision, torch.utils.data
    - datasets (from Hugging Face) for NLP datasets
    - torch_geometric.datasets and ogb.graphproppred for Graph datasets

Usage:
    config_dict = Config().get_config()
    loader = DatasetLoader(config_dict)
    data_dict = loader.load_data("cifar10-noise-0.4-imbalance-50")
    # data_dict contains keys: "train", "val", "test"
"""

import os
import re
import math
import random
import logging
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Subset
import torchvision
import torchvision.transforms as transforms

# For NLP datasets using Hugging Face
try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

# For graph datasets; ogb for ogbg-molhiv and TUDataset from torch_geometric for others.
try:
    from ogb.graphproppred import PygGraphPropPredDataset
except ImportError:
    PygGraphPropPredDataset = None

try:
    from torch_geometric.datasets import TUDataset
except ImportError:
    TUDataset = None

# Configure logging for the module.
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


class DatasetLoader:
    """DatasetLoader loads and preprocesses datasets for CV, NLP, and Graph domains.

    It performs the following:
        - Parses the dataset identifier string to extract base dataset, noise probability, and imbalance factor.
        - Sets seeds for reproducibility across random, numpy, and torch (CPU and GPU).
        - Loads datasets using domain-appropriate libraries and splits the training data into training and validation sets.
        - Applies noise injection (by flipping labels with a given probability) and
         , for CV datasets only, applies imbalance (sub-sampling to create a long-tailed distribution).

    Attributes:
        config (dict): The configuration dictionary loaded from config.yaml.
        seed (int): The seed value chosen from the reproducibility configuration.
    """

    def __init__(self, config: dict) -> None:
        """
        Initializes DatasetLoader with configuration settings.
        Also sets seeds for reproducibility and configures deterministic behavior.

        Args:
            config (dict): The configuration dictionary (from config.yaml).
        """
        self.config = config

        # Set reproducibility seeds using the first seed from config
        seeds = config.get("reproducibility", {}).get("seeds", [])
        if not seeds:
            raise ValueError("No seeds provided in reproducibility configuration.")
        self.seed = int(seeds[0])
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        LOGGER.info(f"DatasetLoader initialized with seed {self.seed}")

    def load_data(self, dataset_identifier: str) -> dict:
        """
        Loads and preprocesses the dataset based on the provided identifier.
        The identifier may include tokens for noise injection (e.g., 'noise-0.4')
        and imbalance (e.g., 'imbalance-50'). The method parses these parameters,
        determines the domain, and calls the corresponding internal loader.

        Args:
            dataset_identifier (str): Identifier for dataset and settings, e.g., "cifar10-noise-0.4-imbalance-50".

        Returns:
            dict: A dictionary with keys "train", "val", and "test" holding the respective datasets.
        """
        lower_id = dataset_identifier.lower()
        noise_p = None
        imbalance_r = None

        # Parse noise setting using regex pattern: look for "noise[-_]?value"
        noise_pattern = r"noise[-_]?(\d+(?:\.\d+)?)"
        noise_match = re.search(noise_pattern, lower_id)
        if noise_match:
            noise_p = float(noise_match.group(1))
            LOGGER.info(f"Detected noise setting: p = {noise_p}")

        # Parse imbalance setting using regex pattern: look for "imbalance[-_]?value"
        imbalance_pattern = r"imbalance[-_]?(\d+)"
        imbalance_match = re.search(imbalance_pattern, lower_id)
        if imbalance_match:
            imbalance_r = int(imbalance_match.group(1))
            LOGGER.info(f"Detected imbalance setting: r = {imbalance_r}")

        # Determine base dataset name.
        dataset_base = None
        if "cifar10" in lower_id:
            dataset_base = "cifar10"
        elif "cifar100" in lower_id:
            dataset_base = "cifar100"
        elif "tinyimagenet" in lower_id or "tiny-imagenet" in lower_id:
            dataset_base = "tinyimagenet"
        elif any(task in lower_id for task in ["rte", "mrpc", "sts-b", "cola", "sst-2", "qnli", "qqp", "mnli"]):
            dataset_base = "glue"
        elif any(name in lower_id for name in ["ogbg-molhiv", "mutag", "proteins", "nci1"]):
            if "ogbg-molhiv" in lower_id:
                dataset_base = "ogbg-molhiv"
            else:
                # For TUDataset types, assume the identifier itself is the dataset name.
                for name in ["mutag", "proteins", "nci1"]:
                    if name in lower_id:
                        dataset_base = name
                        break
        else:
            raise ValueError(f"Unknown dataset identifier: {dataset_identifier}")

        LOGGER.info(f"Base dataset determined: {dataset_base}")

        # Dispatch loading based on domain.
        if dataset_base in ["cifar10", "cifar100", "tinyimagenet"]:
            return self._load_cv_dataset(dataset_base, noise_p, imbalance_r)
        elif dataset_base == "glue":
            return self._load_nlp_dataset(dataset_identifier, noise_p)
        elif dataset_base in ["ogbg-molhiv", "mutag", "proteins", "nci1"]:
            return self._load_graph_dataset(dataset_base, noise_p)
        else:
            raise ValueError(f"Invalid dataset base: {dataset_base}")

    def _load_cv_dataset(self, dataset_base: str, noise_p: float, imbalance_r: int) -> dict:
        """
        Loads and splits CV datasets (CIFAR-10, CIFAR-100, Tiny-ImageNet).

        For CIFAR-10 and CIFAR-100, downloads datasets using torchvision.
        For Tiny-ImageNet, uses ImageFolder assuming standard folder structure.
        The original training set is split into training and validation sets with a 9:1 ratio.
        Noise injection is applied first, then imbalance is applied on the training set if specified.

        Args:
            dataset_base (str): One of "cifar10", "cifar100", or "tinyimagenet".
            noise_p (float): Noise probability (if any).
            imbalance_r (int): Imbalance factor (if any).

        Returns:
            dict: Dictionary with keys "train", "val", "test".
        """
        transform = transforms.ToTensor()
        if dataset_base == "cifar10":
            LOGGER.info("Loading CIFAR-10 dataset...")
            train_full = torchvision.datasets.CIFAR10(
                root="data/cifar10", train=True, download=True, transform=transform
            )
            test_dataset = torchvision.datasets.CIFAR10(
                root="data/cifar10", train=False, download=True, transform=transform
            )
        elif dataset_base == "cifar100":
            LOGGER.info("Loading CIFAR-100 dataset...")
            train_full = torchvision.datasets.CIFAR100(
                root="data/cifar100", train=True, download=True, transform=transform
            )
            test_dataset = torchvision.datasets.CIFAR100(
                root="data/cifar100", train=False, download=True, transform=transform
            )
        elif dataset_base == "tinyimagenet":
            LOGGER.info("Loading Tiny-ImageNet dataset...")
            train_full = torchvision.datasets.ImageFolder(
                root="data/tinyimagenet/train", transform=transform
            )
            test_dataset = torchvision.datasets.ImageFolder(
                root="data/tinyimagenet/val", transform=transform
            )
        else:
            raise ValueError(f"Unsupported CV dataset: {dataset_base}")

        # Apply noise to full training set if specified.
        if noise_p is not None:
            if hasattr(train_full, "classes"):
                candidate_labels = list(range(len(train_full.classes)))
            else:
                candidate_labels = list(set(train_full.targets))
            LOGGER.info("Applying noise injection to training set...")
            train_full = self.apply_noise(train_full, noise_p, candidate_labels)

        # Split training set into train (90%) and validation (10%) splits.
        full_len = len(train_full)
        train_size = int(0.9 * full_len)
        val_size = full_len - train_size
        generator = torch.Generator().manual_seed(self.seed)
        train_subset, val_subset = torch.utils.data.random_split(train_full, [train_size, val_size], generator=generator)
        LOGGER.info(f"Split training set into {train_size} training and {val_size} validation samples.")

        # Apply imbalance to training split if specified.
        if imbalance_r is not None:
            LOGGER.info("Applying imbalance adjustment to training set...")
            train_subset = self.apply_imbalance(train_subset, imbalance_r)

        return {"train": train_subset, "val": val_subset, "test": test_dataset}

    def _load_nlp_dataset(self, dataset_identifier: str, noise_p: float) -> dict:
        """
        Loads NLP datasets from the GLUE benchmark (excluding WNLI).

        The appropriate task is determined from the dataset identifier.
        The train split is modified via noise injection if required.
        The validation split is used as test set since test labels are not provided.

        Args:
            dataset_identifier (str): Identifier containing the GLUE task (e.g., "mrpc-noise-0.4").
            noise_p (float): Noise probability (if any).

        Returns:
            dict: Dictionary with keys "train", "val", "test".
        """
        if load_dataset is None:
            raise ImportError("Hugging Face 'datasets' library is required for NLP datasets.")

        glue_tasks = ["rte", "mrpc", "sts-b", "cola", "sst-2", "qnli", "qqp", "mnli"]
        task_found = None
        lower_id = dataset_identifier.lower()
        for task in glue_tasks:
            if task in lower_id:
                task_found = task
                break
        if task_found is None:
            raise ValueError(f"No valid GLUE task found in dataset identifier '{dataset_identifier}'.")

        LOGGER.info(f"Loading GLUE dataset for task: {task_found}...")
        dataset = load_dataset("glue", task_found)
        # Apply noise to the training split if specified.
        if noise_p is not None:
            candidate_labels = list(range(dataset["train"].features["label"].num_classes))
            LOGGER.info("Applying noise injection to NLP training set...")
            def noise_fn(example):
                if random.random() < noise_p:
                    old_label = example["label"]
                    new_candidates = [lab for lab in candidate_labels if lab != old_label]
                    example["label"] = random.choice(new_candidates)
                return example
            dataset["train"] = dataset["train"].map(noise_fn)
        # Use validation split as test set since test labels are not provided.
        return {"train": dataset["train"], "val": dataset["validation"], "test": dataset["validation"]}

    def _load_graph_dataset(self, dataset_base: str, noise_p: float) -> dict:
        """
        Loads graph datasets. For ogbg-molhiv, uses the official splits provided by OGB.
        For other graph datasets (e.g., MUTAG, PROTEINS, NCI1), loads from TUDataset and
        randomly splits into training, validation, and test sets with an 8:1:1 ratio.
        Noise injection is applied to the training split if specified.

        Args:
            dataset_base (str): Either "ogbg-molhiv" or a TUDataset name (e.g., "mutag").
            noise_p (float): Noise probability (if any).

        Returns:
            dict: Dictionary with keys "train", "val", "test".
        """
        if dataset_base == "ogbg-molhiv":
            if PygGraphPropPredDataset is None:
                raise ImportError("OGB package is required for ogbg-molhiv dataset.")
            LOGGER.info("Loading ogbg-molhiv dataset...")
            dataset = PygGraphPropPredDataset(name="ogbg-molhiv")
            split_idx = dataset.get_idx_split()
            train_idx = split_idx["train"]
            val_idx = split_idx["valid"]
            test_idx = split_idx["test"]
            train_dataset = Subset(dataset, train_idx)
            val_dataset = Subset(dataset, val_idx)
            test_dataset = Subset(dataset, test_idx)
            if noise_p is not None:
                candidate_labels = [0, 1]  # Assuming binary classification.
                LOGGER.info("Applying noise injection to ogbg-molhiv training set...")
                train_dataset = self.apply_noise(train_dataset, noise_p, candidate_labels)
            return {"train": train_dataset, "val": val_dataset, "test": test_dataset}
        else:
            if TUDataset is None:
                raise ImportError("torch_geometric is required for TUDataset.")
            dataset_name = dataset_base.upper()
            LOGGER.info(f"Loading TUDataset: {dataset_name}...")
            dataset = TUDataset(root=f"data/TUDataset/{dataset_name}", name=dataset_name)
            dataset_length = len(dataset)
            train_size = int(0.8 * dataset_length)
            val_size = int(0.1 * dataset_length)
            test_size = dataset_length - train_size - val_size
            generator = torch.Generator().manual_seed(self.seed)
            train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size, test_size], generator=generator
            )
            if noise_p is not None:
                # Determine candidate labels from entire dataset.
                all_labels = [int(dataset[i].y.item()) for i in range(len(dataset))]
                candidate_labels = list(set(all_labels))
                LOGGER.info("Applying noise injection to TUDataset training set...")
                train_dataset = self.apply_noise(train_dataset, noise_p, candidate_labels)
            return {"train": train_dataset, "val": val_dataset, "test": test_dataset}

    def apply_noise(self, dataset: any, p: float, candidate_labels: list) -> any:
        """
        Injects noise into the dataset by flipping labels with probability p.
        This method attempts to support different dataset types:
          - For CV datasets with attribute 'targets' (e.g., CIFAR*), modifies the targets.
          - For NLP datasets (Hugging Face Dataset with 'features'), applies a map to update the 'label'.
          - For graph datasets, iterates via __getitem__ to modify the 'y' attribute.

        Args:
            dataset: The dataset to be modified.
            p (float): The probability of flipping each label.
            candidate_labels (list): List of possible labels.

        Returns:
            The dataset with noise injected.
        """
        if p <= 0:
            return dataset

        # CV dataset handling
        if hasattr(dataset, "targets"):
            if isinstance(dataset, Subset):
                for idx in dataset.indices:
                    old_label = dataset.dataset.targets[idx]
                    if random.random() < p:
                        new_candidates = [lab for lab in candidate_labels if lab != old_label]
                        dataset.dataset.targets[idx] = random.choice(new_candidates)
            else:
                for i in range(len(dataset.targets)):
                    if random.random() < p:
                        old_label = dataset.targets[i]
                        new_candidates = [lab for lab in candidate_labels if lab != old_label]
                        dataset.targets[i] = random.choice(new_candidates)
            return dataset

        # NLP dataset handling (assumes Hugging Face Dataset)
        elif hasattr(dataset, "features") and "label" in dataset.features:
            def noise_fn(example):
                if random.random() < p:
                    old_label = example["label"]
                    new_candidates = [lab for lab in candidate_labels if lab != old_label]
                    example["label"] = random.choice(new_candidates)
                return example
            return dataset.map(noise_fn)

        # Graph dataset handling: iterate over each sample and modify its label if possible.
        elif hasattr(dataset, "__getitem__"):
            for i in range(len(dataset)):
                data = dataset[i]
                if hasattr(data, "y"):
                    old_label = int(data.y.item())
                    if random.random() < p:
                        new_candidates = [lab for lab in candidate_labels if lab != old_label]
                        data.y = torch.tensor([random.choice(new_candidates)], dtype=data.y.dtype)
            return dataset

        else:
            LOGGER.warning("apply_noise: Unsupported dataset type for noise injection. Skipping noise injection.")
            return dataset

    def apply_imbalance(self, dataset: any, r: int) -> any:
        """
        Reduces the dataset to create a long-tailed imbalance distribution for CV datasets only.

        For a balanced dataset, groups sample indices by label and selects a random subset
        from each group based on the exponential decay rule:
            n_c = n_ref * d^c, where d = (1/r)^(1/(C-1)),
        and n_ref is chosen as the minimum number of samples among all classes.
        A new torch.utils.data.Subset is returned with only the selected indices.

        Args:
            dataset: The dataset (or Subset) with a 'targets' attribute.
            r (int): The imbalance factor (ratio between the largest and smallest class counts).

        Returns:
            A torch.utils.data.Subset representing the imbalanced dataset.
        """
        # Retrieve the underlying dataset and indices.
        if isinstance(dataset, Subset):
            base_dataset = dataset.dataset
            indices = list(dataset.indices)
        else:
            base_dataset = dataset
            indices = list(range(len(dataset)))

        if not hasattr(base_dataset, "targets"):
            LOGGER.warning("apply_imbalance: Dataset does not have 'targets' attribute. Skipping imbalance adjustment.")
            return dataset

        # Extract targets for provided indices.
        targets = [base_dataset.targets[i] for i in indices]

        # Group indices by class.
        class_indices = defaultdict(list)
        for idx, label in zip(indices, targets):
            class_indices[label].append(idx)

        sorted_classes = sorted(class_indices.keys())
        counts = [len(class_indices[c]) for c in sorted_classes]
        n_ref = min(counts)
        if n_ref == 0:
            LOGGER.warning("apply_imbalance: One of the classes has 0 samples. Skipping imbalance adjustment.")
            return dataset

        # Compute decay factor: d = (1/r)^(1/(C-1))
        C = len(sorted_classes)
        if C <= 1:
            LOGGER.warning("apply_imbalance: Only one class found. Skipping imbalance adjustment.")
            return dataset
        d = (1 / r) ** (1 / (C - 1))

        new_indices = []
        # For each class (in sorted order), select a weighted subset of indices.
        for i, cls in enumerate(sorted_classes):
            target_count = math.floor(n_ref * (d ** i))
            target_count = max(1, target_count)  # Ensure at least one sample is selected.
            available = class_indices[cls]
            # Set seed for reproducible selection; vary per class.
            random.seed(self.seed + cls)
            if len(available) <= target_count:
                selected = available
            else:
                selected = random.sample(available, target_count)
            new_indices.extend(selected)

        LOGGER.info(f"apply_imbalance: Reduced dataset from {len(indices)} to {len(new_indices)} samples.")
        return Subset(base_dataset, new_indices)
