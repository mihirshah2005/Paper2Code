"""
dataset_loader.py

This module defines the DatasetLoader class for loading and preprocessing datasets
across three domains: Computer Vision (CV), Natural Language Processing (NLP), and Graph.
It supports splitting datasets into train, validation, and test sets, and applies experimental
preprocessing such as label noise injection and class imbalance creation as specified by
the dataset identifier (e.g., "cifar10-noise-0.4" or "tinyimagenet-imbalance-50").

The class uses configuration parameters loaded from "config.yaml" (via config.py) and ensures
that default values and explicit type checking are used throughout.
"""

import os
import logging
import numpy as np
import torch
from torch.utils.data import random_split, Subset
from typing import Any, Dict, List, Optional, Tuple

# For computer vision datasets
import torchvision
from torchvision import transforms

# For NLP datasets: Using Hugging Face datasets package
try:
    from datasets import load_dataset
except ImportError:
    raise ImportError("Please install the 'datasets' package from Hugging Face for NLP functionality.")

# For graph datasets
try:
    from torch_geometric.datasets import TUDataset
except ImportError:
    raise ImportError("Please install 'torch-geometric' package for Graph datasets functionality.")
try:
    from ogb.graphproppred import PygGraphPropPredDataset
except ImportError:
    # If ogb is not available, we simply warn. OGB is required for ogbg-molhiv.
    PygGraphPropPredDataset = None
    logging.warning("OGB package not found. 'ogbg-molhiv' dataset will not be available.")

# Configure module level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class DatasetLoader:
    """
    DatasetLoader loads raw datasets based on the provided configuration and dataset identifier.
    It supports domains: CV, NLP, and graph. The identifier string (e.g., "cifar10-noise-0.4")
    indicates the base dataset and any experimental modifications (noise, imbalance) to apply.
    
    Public Methods:
        - load_data() -> Dict[str, Any]: Loads and preprocesses the dataset and returns a dictionary 
          with keys "train", "val", and "test".
        - apply_noise(data: Any, p: float) -> Any: Injects noise into the training labels.
        - apply_imbalance(data: Any, r: int) -> Any: Creates a long-tailed distribution for CV datasets.
    """

    def __init__(self, config: Dict[str, Any], dataset_id: str = "cifar10") -> None:
        """
        Initialize the DatasetLoader with configuration and dataset identifier.
        :param config: Configuration dictionary loaded from config.yaml.
        :param dataset_id: String identifier of the dataset with optional modifiers (e.g., "cifar10-noise-0.4").
        """
        self.config: Dict[str, Any] = config
        self.dataset_id: str = dataset_id.strip().lower()
        self.core_name: str
        self.noise_prob: Optional[float] = None
        self.imbalance_factor: Optional[int] = None
        self.domain: str = ""
        self._parse_dataset_identifier(self.dataset_id)
        
        # Set global reproducibility seed from configuration (use first seed as default)
        reproducibility_conf = self.config.get("reproducibility", {})
        seeds: List[int] = reproducibility_conf.get("seeds", [42])
        self.global_seed: int = int(seeds[0])
        self.random_state = np.random.RandomState(self.global_seed)

        # Determine domain based on core_name
        cv_datasets = ["cifar10", "cifar100", "tinyimagenet"]
        nlp_datasets = ["rte", "mrpc", "sts-b", "cola", "sst-2", "qnli", "qqp", "mnli"]
        graph_datasets = ["ogbg-molhiv", "mutag", "proteins", "nci1"]

        if self.core_name in cv_datasets:
            self.domain = "cv"
        elif self.core_name in nlp_datasets:
            self.domain = "nlp"
        elif self.core_name in graph_datasets:
            self.domain = "graph"
        else:
            error_msg = f"Dataset '{self.core_name}' not recognized in any supported domain."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info(f"Initialized DatasetLoader for domain '{self.domain}' with dataset '{self.core_name}', "
                    f"noise probability: {self.noise_prob}, imbalance factor: {self.imbalance_factor}.")

    def _parse_dataset_identifier(self, dataset_id: str) -> None:
        """
        Parses the dataset identifier string to extract the base dataset name, noise probability,
        and imbalance factor if specified.
        Expected formats:
            - "cifar10" (standard)
            - "cifar10-noise-0.4"
            - "tinyimagenet-imbalance-50"
            - Combination (order insensitive): "cifar100-noise-0.2-imbalance-50"
        Sets instance variables: self.core_name, self.noise_prob, self.imbalance_factor.
        :param dataset_id: The dataset identifier string.
        :return: None.
        :raises: ValueError if required modifiers are missing their parameters.
        """
        tokens: List[str] = dataset_id.split('-')
        if len(tokens) < 1:
            raise ValueError("Invalid dataset identifier provided.")
        
        self.core_name = tokens[0]
        i: int = 1
        while i < len(tokens):
            token = tokens[i]
            if token == "noise":
                if i + 1 < len(tokens):
                    try:
                        self.noise_prob = float(tokens[i + 1])
                    except ValueError:
                        raise ValueError("Invalid noise probability value in dataset identifier.")
                    i += 2
                else:
                    raise ValueError("Noise modifier specified without a probability value.")
            elif token == "imbalance":
                if i + 1 < len(tokens):
                    try:
                        self.imbalance_factor = int(tokens[i + 1])
                    except ValueError:
                        raise ValueError("Invalid imbalance factor in dataset identifier.")
                    i += 2
                else:
                    raise ValueError("Imbalance modifier specified without a factor value.")
            else:
                i += 1

    def load_data(self) -> Dict[str, Any]:
        """
        Loads and preprocesses the dataset based on the domain and identifier.
        Applies the train/validation/test split and calls apply_noise and apply_imbalance if required.
        :return: Dictionary with keys: "train", "val", and "test".
        """
        if self.domain == "cv":
            return self.__load_cv_data()
        elif self.domain == "nlp":
            return self.__load_nlp_data()
        elif self.domain == "graph":
            return self.__load_graph_data()
        else:
            error_msg = f"Unsupported domain '{self.domain}' encountered in load_data()."
            logger.error(error_msg)
            raise ValueError(error_msg)

    def __load_cv_data(self) -> Dict[str, Any]:
        """
        Load Computer Vision datasets (CIFAR-10, CIFAR-100, Tiny-ImageNet) using torchvision.
        Splits training data into train and validation in a 9:1 ratio.
        Applies noise and imbalance modifications if specified.
        :return: Dictionary with keys "train", "val", "test".
        """
        transform = transforms.Compose([transforms.ToTensor()])
        generator = torch.Generator().manual_seed(self.global_seed)

        if self.core_name == "cifar10":
            logger.info("Loading CIFAR-10 dataset.")
            full_train = torchvision.datasets.CIFAR10(root="./data/cifar10", train=True, download=True, transform=transform)
            test_set = torchvision.datasets.CIFAR10(root="./data/cifar10", train=False, download=True, transform=transform)
        elif self.core_name == "cifar100":
            logger.info("Loading CIFAR-100 dataset.")
            full_train = torchvision.datasets.CIFAR100(root="./data/cifar100", train=True, download=True, transform=transform)
            test_set = torchvision.datasets.CIFAR100(root="./data/cifar100", train=False, download=True, transform=transform)
        elif self.core_name == "tinyimagenet":
            logger.info("Loading Tiny-ImageNet dataset.")
            # Expect Tiny-ImageNet downloaded and extracted to "./data/tinyimagenet"
            train_dir = os.path.join("./data", "tinyimagenet", "train")
            val_dir = os.path.join("./data", "tinyimagenet", "val")
            if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
                error_msg = f"Directories for Tiny-ImageNet not found in expected locations: {train_dir} and {val_dir}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            full_train = torchvision.datasets.ImageFolder(root=train_dir, transform=transform)
            test_set = torchvision.datasets.ImageFolder(root=val_dir, transform=transform)
        else:
            error_msg = f"Unsupported CV dataset: {self.core_name}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Split training set into train (90%) and validation (10%)
        total_samples: int = len(full_train)
        train_len: int = int(0.9 * total_samples)
        val_len: int = total_samples - train_len
        train_set, val_set = random_split(full_train, [train_len, val_len], generator=generator)
        logger.info(f"Split dataset into {train_len} training and {val_len} validation samples.")

        # Apply noise injection if specified
        if self.noise_prob is not None:
            logger.info(f"Applying noise injection with probability {self.noise_prob} to training set.")
            train_set = self.apply_noise(train_set, self.noise_prob)

        # Apply imbalance if specified and only for CV datasets
        if self.imbalance_factor is not None and self.domain == "cv":
            logger.info(f"Applying imbalance with factor {self.imbalance_factor} to training set.")
            train_set = self.apply_imbalance(train_set, self.imbalance_factor)

        return {"train": train_set, "val": val_set, "test": test_set}

    def __load_nlp_data(self) -> Dict[str, Any]:
        """
        Load NLP datasets from the GLUE benchmark using Hugging Face's load_dataset.
        Excludes problematic datasets (e.g., WNLI). Uses the validation set as the test set.
        Applies noise injection to training set if specified.
        :return: Dictionary with keys "train", "val", "test".
        """
        if self.core_name == "wnli":
            error_msg = "WNLI is excluded due to known issues."
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"Loading GLUE dataset for task '{self.core_name}'.")
        try:
            dataset = load_dataset("glue", self.core_name)
        except Exception as e:
            error_msg = f"Failed to load GLUE dataset for task '{self.core_name}': {e}"
            logger.error(error_msg)
            raise e

        # Use the 'train' split as training and 'validation' split as both validation and test.
        train_data = dataset["train"]
        if "validation" in dataset:
            val_data = dataset["validation"]
            test_data = dataset["validation"]
        else:
            error_msg = f"Validation split not found in the GLUE dataset for '{self.core_name}'."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Convert train_data to a list of dictionaries if noise injection is required because
        # Hugging Face Datasets are immutable.
        if self.noise_prob is not None:
            logger.info(f"Applying noise injection with probability {self.noise_prob} to NLP training set.")
            train_data_list = list(train_data)
            train_data_list = self.apply_noise(train_data_list, self.noise_prob)
            train_data = train_data_list

        # Imbalance is not applied for NLP datasets.
        return {"train": train_data, "val": val_data, "test": test_data}

    def __load_graph_data(self) -> Dict[str, Any]:
        """
        Load Graph datasets. For ogbg-molhiv, uses the official split provided in the dataset.
        For TUDataset based datasets (e.g., MUTAG, PROTEINS, NCI1), randomly splits the dataset in an 8:1:1 ratio.
        Applies noise injection to training set if specified.
        :return: Dictionary with keys "train", "val", "test".
        """
        if self.core_name == "ogbg-molhiv":
            if PygGraphPropPredDataset is None:
                error_msg = "OGB package not available. Cannot load 'ogbg-molhiv'."
                logger.error(error_msg)
                raise ImportError(error_msg)
            logger.info("Loading ogbg-molhiv dataset.")
            dataset = PygGraphPropPredDataset(name="ogbg-molhiv", root="./data/ogbg")
            split_idx = dataset.get_idx_split()
            train_set = dataset[split_idx["train"]]
            val_set = dataset[split_idx["valid"]]
            test_set = dataset[split_idx["test"]]
        else:
            # Assume dataset is one of the TUDataset datasets: e.g., MUTAG, PROTEINS, NCI1.
            logger.info(f"Loading TUDataset for '{self.core_name.upper()}'.")
            dataset = TUDataset(root=os.path.join("./data", "TUDataset"), name=self.core_name.upper())
            total_samples: int = len(dataset)
            train_len: int = int(0.8 * total_samples)
            rem_len: int = total_samples - train_len
            val_len: int = rem_len // 2
            test_len: int = rem_len - val_len
            generator = torch.Generator().manual_seed(self.global_seed)
            train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len], generator=generator)

        # Apply noise injection if specified (imbalance is not applied in graph domain)
        if self.noise_prob is not None:
            logger.info(f"Applying noise injection with probability {self.noise_prob} to graph training set.")
            train_set = self.apply_noise(train_set, self.noise_prob)

        return {"train": train_set, "val": val_set, "test": test_set}

    def apply_noise(self, data: Any, p: float) -> Any:
        """
        Applies label noise to the provided dataset. For each sample in the dataset,
        with probability p, the original label is replaced by a random label drawn uniformly
        from the set of available classes (excluding the original label).
        Supports multiple dataset types:
            - For torchvision datasets with a "targets" attribute.
            - For Hugging Face NLP datasets provided as a list of dicts with a "label" key.
            - For graph datasets where each data object has attribute "y".
        :param data: The dataset to apply noise to.
        :param p: The noise probability (0.0 <= p <= 1.0).
        :return: The dataset with noisy labels.
        :raises: ValueError if p is out of range.
        """
        if not (0.0 <= p <= 1.0):
            error_msg = f"Noise probability {p} is out of the valid range [0, 1]."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Define a helper function to select a random label different from the original.
        def random_label(original: int, available_labels: List[int]) -> int:
            choices = [lbl for lbl in available_labels if lbl != original]
            return int(self.random_state.choice(choices, 1)[0])

        # For torchvision datasets (CV) that have 'targets' attribute.
        if hasattr(data, "targets"):
            available_labels = sorted(list(set(data.targets)))
            noisy_targets = []
            for orig in data.targets:
                if self.random_state.rand() < p:
                    noisy_targets.append(random_label(orig, available_labels))
                else:
                    noisy_targets.append(orig)
            data.targets = noisy_targets
            logger.info("Noise injection applied to 'targets' attribute of the dataset.")
            return data

        # For Hugging Face NLP datasets (assumed to be a list of dicts with key "label")
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict) and "label" in data[0]:
            # Determine available labels from the dataset.
            all_labels = [sample["label"] for sample in data]
            available_labels = sorted(list(set(all_labels)))
            for sample in data:
                if self.random_state.rand() < p:
                    sample["label"] = random_label(sample["label"], available_labels)
            logger.info("Noise injection applied to NLP dataset labels.")
            return data

        # For graph datasets: assume data is indexable and each sample has attribute 'y'
        try:
            # Try iterating over indices to update each sample
            for idx in range(len(data)):
                sample = data[idx]
                if hasattr(sample, "y"):
                    # Get available labels from the dataset if possible; otherwise, infer from current labels.
                    if hasattr(data, "num_classes"):
                        available_labels = list(range(int(data.num_classes)))
                    else:
                        # Fallback: infer available labels from current sample (not ideal)
                        available_labels = [int(sample.y.item())]
                    # Only change if p condition met.
                    if self.random_state.rand() < p:
                        new_label = random_label(int(sample.y.item()), available_labels)
                        sample.y = torch.tensor([new_label])
            logger.info("Noise injection applied to graph dataset labels ('y' attribute).")
            return data
        except Exception as e:
            logger.error(f"Failed to apply noise injection: {e}")
            raise e

    def apply_imbalance(self, data: Any, r: int) -> Any:
        """
        Applies class imbalance to a CV dataset by resampling the training set based on an exponential
        decay function.
        This function is only applicable to CV datasets. For other domains, the dataset is returned unchanged.
        :param data: The dataset (assumed to have a "targets" attribute).
        :param r: Imbalance factor (integer greater than or equal to 1). r == 1 implies no imbalance.
        :return: The imbalanced dataset wrapped in a torch.utils.data.Subset.
        :raises: ValueError if r is less than 1.
        """
        if self.domain != "cv":
            logger.warning("apply_imbalance is only applicable to CV datasets. Returning original dataset.")
            return data

        if r < 1:
            error_msg = f"Imbalance factor r must be >= 1. Got {r}."
            logger.error(error_msg)
            raise ValueError(error_msg)
        if r == 1:
            logger.info("Imbalance factor is 1 (balanced). Returning original dataset.")
            return data

        # Ensure the dataset has 'targets' attribute.
        if not hasattr(data, "targets"):
            error_msg = "Dataset does not have 'targets' attribute required for imbalance operation."
            logger.error(error_msg)
            raise AttributeError(error_msg)

        # Group indices by class label.
        targets: List[int] = data.targets
        unique_classes = sorted(list(set(targets)))
        num_classes: int = len(unique_classes)
        # Assume original dataset is balanced; n0 is count for class 0.
        # It is safe to use the first class in sorted order.
        n0: int = targets.count(unique_classes[0])
        # Compute decay factor d using the formula: d = (1/r)^(1/(C-1))
        d: float = (1.0 / r) ** (1.0 / (num_classes - 1))
        logger.info(f"Calculated decay factor d = {d} for {num_classes} classes with r = {r}.")
        indices_to_keep: List[int] = []

        # For each class, determine target sample count and randomly select indices.
        for class_idx, cls in enumerate(unique_classes):
            class_indices = [i for i, label in enumerate(targets) if label == cls]
            target_count: int = max(1, int(round(n0 * (d ** class_idx))))
            if target_count > len(class_indices):
                target_count = len(class_indices)
            selected = self.random_state.choice(class_indices, size=target_count, replace=False).tolist()
            indices_to_keep.extend(selected)
            logger.info(f"Class {cls}: original count = {len(class_indices)}, target count = {target_count}.")

        # Return a Subset of the original dataset containing only the selected indices.
        logger.info(f"Total samples after imbalance: {len(indices_to_keep)}.")
        return Subset(data, indices_to_keep)
