"""dataset_loader.py

This module defines the DatasetLoader class responsible for loading and preprocessing
the image dataset (e.g., ImageNet) with aspect-ratio preserving resizing, horizontal
flip augmentation, and metadata generation for further tokenization.

The preprocessing includes:
    - Resizing images to ensure that the area does not exceed a maximum (default 256x256 = 65536),
      while preserving the original aspect ratio.
    - Applying a horizontal flip augmentation with 50% probability.
    - Converting images to PyTorch tensors.
    - Computing metadata: original dimensions, resized dimensions, nominal patch size, and
      estimated token counts based on floor division.

All configuration parameters are read from the provided config dictionary.
"""

import os
import math
import random
from typing import Any, Dict, List
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CustomImageDataset(Dataset):
    """
    A custom PyTorch Dataset that loads images from a specified directory,
    applies a sequence of transformations, and provides accompanying metadata
    for downstream tokenization and VAE encoding.
    """

    def __init__(self, dataset_dir: str, config: Dict[str, Any]) -> None:
        """
        Initialize the CustomImageDataset.

        Args:
            dataset_dir (str): Path to the directory containing images.
            config (Dict[str, Any]): Configuration dictionary loaded from config.yaml.
        """
        self.dataset_dir: str = dataset_dir
        self.config: Dict[str, Any] = config

        # Allowed image file extensions.
        allowed_extensions: List[str] = [".jpg", ".jpeg", ".png", ".bmp"]
        self.image_paths: List[str] = []

        # Walk through the dataset directory and collect image file paths.
        for root, _, files in os.walk(self.dataset_dir):
            for file in files:
                ext: str = os.path.splitext(file)[1].lower()
                if ext in allowed_extensions:
                    self.image_paths.append(os.path.join(root, file))
        self.image_paths.sort()  # Ensure reproducibility

        # Transformation to convert PIL Image to tensor.
        self.to_tensor = transforms.ToTensor()

        # Fetch preprocessing configurations.
        self.resize_max_area: int = self.config.get("data", {}).get("resize_max_area", 65536)
        self.augmentation: str = self.config.get("data", {}).get("augmentation", "Horizontal Flip")
        self.patch_size: int = self.config.get("model", {}).get("patch_size", 2)

    def __len__(self) -> int:
        """
        Return the total number of images.
        """
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Retrieve and preprocess an image along with its metadata.

        The preprocessing steps include:
            - Resizing (if necessary) to ensure the image area is <= resize_max_area.
            - Horizontal flip augmentation based on the configuration.
            - Conversion to a PyTorch tensor.
            - Calculation of estimated token counts based on the patch size.

        Args:
            index (int): Index of the image in the dataset.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - "image": A tensor of shape (C, H, W).
                - "metadata": Metadata including original and resized dimensions,
                              patch size, and estimated token counts.
        """
        image_path: str = self.image_paths[index]
        try:
            image: Image.Image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise IOError(f"Error opening image {image_path}: {e}")

        # Retrieve original dimensions.
        original_width: int
        original_height: int
        original_width, original_height = image.size

        # Compute new dimensions to ensure that image area does not exceed resize_max_area.
        original_area: int = original_width * original_height
        if original_area > self.resize_max_area:
            scale_factor: float = math.sqrt(self.resize_max_area / original_area)
            new_width: int = int(round(original_width * scale_factor))
            new_height: int = int(round(original_height * scale_factor))
        else:
            new_width = original_width
            new_height = original_height

        # Resize the image using bilinear interpolation.
        image = image.resize((new_width, new_height), resample=Image.BILINEAR)

        # Apply horizontal flip augmentation if specified in the configuration.
        if self.augmentation == "Horizontal Flip":
            if random.random() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # Convert the processed image to a tensor.
        image_tensor: torch.Tensor = self.to_tensor(image)

        # Estimate token counts based on the patch size using floor division.
        est_tokens_height: int = new_height // self.patch_size
        est_tokens_width: int = new_width // self.patch_size
        estimated_token_count: int = est_tokens_height * est_tokens_width

        metadata: Dict[str, Any] = {
            "original_width": original_width,
            "original_height": original_height,
            "resized_width": new_width,
            "resized_height": new_height,
            "patch_size": self.patch_size,
            "estimated_tokens_height": est_tokens_height,
            "estimated_tokens_width": est_tokens_width,
            "estimated_token_count": estimated_token_count,
        }

        return {"image": image_tensor, "metadata": metadata}


class DatasetLoader:
    """
    DatasetLoader is responsible for initializing and loading the image dataset as a DataLoader.
    
    It constructs the CustomImageDataset with the appropriate image pre-processing pipeline
    and wraps it in a PyTorch DataLoader for use in downstream training modules.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the DatasetLoader with the given configuration.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary loaded from config.yaml.
        """
        self.config: Dict[str, Any] = config
        # Use the dataset path from the configuration; default to "./data/imagenet" if not provided.
        self.dataset_dir: str = self.config.get("data", {}).get("dataset_path", "./data/imagenet")

    def load_data(self) -> DataLoader:
        """
        Load the dataset and return a PyTorch DataLoader.

        Returns:
            DataLoader: A DataLoader wrapping the CustomImageDataset.
        """
        dataset: CustomImageDataset = CustomImageDataset(self.dataset_dir, self.config)
        batch_size: int = self.config.get("training", {}).get("batch_size", 256)
        data_loader: DataLoader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        return data_loader


# For testing purposes only: This block allows the module to be run as a script.
if __name__ == "__main__":
    import yaml

    CONFIG_PATH: str = "config.yaml"
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as config_file:
            config: Dict[str, Any] = yaml.safe_load(config_file)
    else:
        # Default configuration if config.yaml is not found.
        config = {
            "training": {"batch_size": 256},
            "data": {
                "resize_max_area": 65536,
                "augmentation": "Horizontal Flip",
                "dataset_path": "./data/imagenet"
            },
            "model": {"patch_size": 2}
        }
    dataset_loader_instance: DatasetLoader = DatasetLoader(config)
    dataloader: DataLoader = dataset_loader_instance.load_data()
    print(f"Total number of batches: {len(dataloader)}")
    for batch in dataloader:
        print("Batch metadata:", batch["metadata"])
        break
