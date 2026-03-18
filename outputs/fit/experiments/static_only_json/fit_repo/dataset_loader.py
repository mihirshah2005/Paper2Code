"""dataset_loader.py

This module defines the DatasetLoader class which is responsible for loading an image dataset
(e.g., ImageNet) using ImageFolder, applying appropriate transformations (resizing while preserving
aspect ratio, horizontal flip augmentation, conversion to tensor), and computing metadata needed for
the latent tokenization process. This includes calculating the dynamic token length and generating
a fixed-length token mask based on the configuration.
"""

import os
import math
import logging
from typing import Any, Callable, Dict, Tuple, List

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image

# Setup module-level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ResizePreserveAspect:
    """Custom resize transform that preserves the aspect ratio and ensures that the image area
    does not exceed a specified maximum (max_area)."""

    def __init__(self, max_area: int) -> None:
        """
        Args:
            max_area (int): The maximum allowed image area (width x height).
        """
        self.max_area = max_area

    def __call__(self, image: Image.Image) -> Image.Image:
        """
        Resize the image if its area exceeds self.max_area; otherwise, return the original image.

        Args:
            image (PIL.Image.Image): The input image.

        Returns:
            PIL.Image.Image: The resized image (if necessary) with preserved aspect ratio.
        """
        original_width, original_height = image.size
        area = original_width * original_height
        if area > self.max_area:
            scale_factor = math.sqrt(self.max_area / area)
            new_width = max(1, int(round(original_width * scale_factor)))
            new_height = max(1, int(round(original_height * scale_factor)))
            resized_image = image.resize((new_width, new_height), resample=Image.BILINEAR)
            return resized_image
        return image


def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function to batch variable-sized image tensors and fixed-size token masks.

    Args:
        batch (List[Dict[str, Any]]): A list of dictionaries from the dataset __getitem__.

    Returns:
        Dict[str, Any]: A dictionary containing batched items.
            - "images": List of image tensors (variable sizes).
            - "token_counts": Tensor of token counts (shape [batch_size]).
            - "token_masks": Tensor of token masks (shape [batch_size, max_token_length]).
            - "original_sizes": List of tuples (H, W) for each sample.
    """
    images = [item["image"] for item in batch]
    token_counts = torch.tensor([item["token_count"] for item in batch], dtype=torch.int64)
    token_masks = torch.stack([item["token_mask"] for item in batch])
    original_sizes = [item["original_size"] for item in batch]
    return {
        "images": images,
        "token_counts": token_counts,
        "token_masks": token_masks,
        "original_sizes": original_sizes,
    }


class CustomImageDataset(Dataset):
    """Custom dataset that wraps torchvision.datasets.ImageFolder and computes latent tokenization metadata."""

    def __init__(
        self,
        root: str,
        transform: Callable,
        patch_size: int,
        max_token_length: int,
        downsample_factor: int = 8,
    ) -> None:
        """
        Args:
            root (str): Root directory of the image dataset.
            transform (Callable): Composed transformation to apply on each image.
            patch_size (int): Patch size used for patchifying VAE latent outputs.
            max_token_length (int): Fixed maximum token length for padding token sequences.
            downsample_factor (int, optional): Factor by which the pretrained VAE downsamples images.
                                               Default is 8 as in Stable Diffusion.
        """
        self.root = root
        self.transform = transform
        self.patch_size = patch_size
        self.max_token_length = max_token_length
        self.downsample_factor = downsample_factor

        # Use ImageFolder to load images assuming images are organized in class folders.
        if not os.path.exists(self.root):
            logger.warning("Dataset root '%s' does not exist. Please check the path.", self.root)
        self.data = datasets.ImageFolder(root=self.root, transform=self.transform)

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Retrieve an image and compute tokenization metadata based on its size.

        Args:
            index (int): The index of the sample.

        Returns:
            Dict[str, Any]: A dictionary with keys:
                - "image": Transformed image tensor.
                - "token_count": Computed number of valid tokens (after patchification).
                - "token_mask": Tensor of shape (max_token_length,) with valid tokens as 1 and padding as 0.
                - "original_size": Tuple (H, W) of the post-transform image dimensions.
        """
        try:
            # Get image and label from the ImageFolder dataset; label is not used.
            image, _ = self.data[index]
        except Exception as e:
            logger.error("Error loading image at index %d: %s", index, str(e))
            raise e

        # image is a tensor of shape (C, H, W)
        if not torch.is_tensor(image):
            logger.error("Transformed image is not a torch.Tensor at index %d.", index)
            raise TypeError("Expected image to be a torch.Tensor.")

        _, height, width = image.shape
        original_size: Tuple[int, int] = (height, width)

        # Compute latent dimensions based on the VAE downsampling factor.
        latent_height: int = math.floor(height / self.downsample_factor)
        latent_width: int = math.floor(width / self.downsample_factor)

        # Compute token dimensions after patchification.
        token_height: int = math.floor(latent_height / self.patch_size)
        token_width: int = math.floor(latent_width / self.patch_size)
        token_count: int = token_height * token_width

        # If token_count exceeds max_token_length, clip it.
        if token_count > self.max_token_length:
            logger.warning(
                "Computed token_count (%d) exceeds max_token_length (%d) for index %d. Clipping token_count.",
                token_count, self.max_token_length, index
            )
            token_count = self.max_token_length

        # Create a fixed-length token mask: first token_count entries are 1, rest are 0.
        token_mask_list: List[int] = [1] * token_count + [0] * (self.max_token_length - token_count)
        token_mask: torch.Tensor = torch.tensor(token_mask_list, dtype=torch.int64)

        return {
            "image": image,
            "token_count": token_count,
            "token_mask": token_mask,
            "original_size": original_size,
        }


class DatasetLoader:
    """Loader class for the image dataset. It wraps the custom dataset and provides a DataLoader."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Args:
            config (Dict[str, Any]): Configuration dictionary (from config.yaml) with dataset and model parameters.
        """
        # Data configuration
        data_config: Dict[str, Any] = config.get("data", {})
        self.dataset_name: str = data_config.get("dataset", "ImageNet")
        self.resize_max_area: int = int(data_config.get("resize_max_area", 65536))
        augmentation: str = data_config.get("augmentation", "Horizontal Flip")
        dataset_root: str = data_config.get("root", "./data")

        # Model configuration
        model_config: Dict[str, Any] = config.get("model", {})
        self.patch_size: int = int(model_config.get("patch_size", 2))
        self.max_token_length: int = int(model_config.get("max_token_length", 256))

        # Training configuration
        training_config: Dict[str, Any] = config.get("training", {})
        self.batch_size: int = int(training_config.get("batch_size", 256))

        # Assumed constant for VAE downsampling factor (as in Stable Diffusion)
        self.downsample_factor: int = 8

        # Build transformation pipeline.
        transform_list: List[Callable] = []
        # Custom resizing preserving aspect ratio.
        transform_list.append(ResizePreserveAspect(self.resize_max_area))
        # Data augmentation: apply horizontal flip if specified.
        if augmentation.lower() == "horizontal flip":
            transform_list.append(transforms.RandomHorizontalFlip())
        # Convert image to tensor.
        transform_list.append(transforms.ToTensor())
        self.transform: transforms.Compose = transforms.Compose(transform_list)

        # Initialize the custom dataset.
        self.dataset: Dataset = CustomImageDataset(
            root=dataset_root,
            transform=self.transform,
            patch_size=self.patch_size,
            max_token_length=self.max_token_length,
            downsample_factor=self.downsample_factor
        )

    def __len__(self) -> int:
        """
        Returns:
            int: Total number of images in the dataset.
        """
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Proxy to the custom dataset's __getitem__ method.

        Args:
            index (int): Index of the sample.

        Returns:
            Dict[str, Any]: Data sample dictionary.
        """
        return self.dataset[index]

    def load_data(self) -> DataLoader:
        """
        Wrap the custom dataset in a DataLoader.

        Returns:
            DataLoader: A PyTorch DataLoader with the configured batch size, shuffling enabled, 
                        and a custom collate function to handle variable image sizes.
        """
        dataloader: DataLoader = DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=custom_collate_fn,
            num_workers=4,  # You can adjust this number as needed.
            pin_memory=True
        )
        return dataloader
