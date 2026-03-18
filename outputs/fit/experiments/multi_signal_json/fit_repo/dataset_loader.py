## dataset_loader.py
import os
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image


def resize_preserve_aspect(image: Image.Image, max_area: int) -> Image.Image:
    """
    Resize an image to ensure its area does not exceed max_area while preserving aspect ratio.
    If the image area is less than or equal to max_area, returns the image unchanged.
    
    Args:
        image (PIL.Image.Image): The input image.
        max_area (int): Maximum allowed area (width * height).

    Returns:
        PIL.Image.Image: The resized (or original) image.
    """
    width, height = image.size
    area = width * height
    if area > max_area:
        scale = math.sqrt(max_area / float(area))
        new_width = max(1, int(math.floor(width * scale)))
        new_height = max(1, int(math.floor(height * scale)))
        return image.resize((new_width, new_height), Image.BICUBIC)
    return image


def patchify_and_pad(
    latent_tensor: torch.Tensor, patch_size: int, max_token_length: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a latent tensor into a sequence of flattened patch tokens and pad/truncate 
    the sequence to a fixed length. The ordering is row-major:
        - Extract patches row by row from top-left to bottom-right.
        - Each patch is flattened in row-major order.
    
    Args:
        latent_tensor (torch.Tensor): Input tensor of shape [C, H, W] from the VAE encoder.
        patch_size (int): Size of each patch (patch is patch_size x patch_size).
        max_token_length (int): Maximum allowed token sequence length.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - tokens: Tensor of shape [max_token_length, token_dim] where token_dim = C * patch_size^2.
            - attention_mask: Tensor of shape [max_token_length] with 1.0 for real tokens and 0.0 for padded tokens.
    """
    # Ensure latent_tensor has 3 dimensions: [C, H, W]
    if latent_tensor.dim() != 3:
        raise ValueError("latent_tensor must be a 3D tensor of shape [C, H, W].")
    
    C, H, W = latent_tensor.shape
    # Calculate required padding if dimensions are not divisible by patch_size
    pad_h = (patch_size - (H % patch_size)) % patch_size
    pad_w = (patch_size - (W % patch_size)) % patch_size
    if pad_h > 0 or pad_w > 0:
        # F.pad expects pad in the order: (pad_left, pad_right, pad_top, pad_bottom)
        latent_tensor = F.pad(latent_tensor, (0, pad_w, 0, pad_h), mode="constant", value=0)
        H += pad_h
        W += pad_w

    # Use unfold to extract non-overlapping patches; input shape must be [B, C, H, W]
    latent_tensor_unsq = latent_tensor.unsqueeze(0)  # Shape: [1, C, H, W]
    patch_extractor = nn.Unfold(kernel_size=patch_size, stride=patch_size)
    patches = patch_extractor(latent_tensor_unsq)  # Shape: [1, C*(patch_size**2), L]
    patches = patches.squeeze(0).transpose(0, 1)     # Shape: [L, token_dim]
    token_dim = patches.shape[1]
    orig_token_count = patches.shape[0]

    # Truncate if token count exceeds max_token_length
    if orig_token_count > max_token_length:
        tokens = patches[:max_token_length, :]
        real_token_count = max_token_length
    else:
        tokens = patches
        real_token_count = orig_token_count
        # Pad the remaining tokens with zeros
        if real_token_count < max_token_length:
            pad_tokens = torch.zeros(
                (max_token_length - real_token_count, token_dim),
                dtype=tokens.dtype,
                device=tokens.device,
            )
            tokens = torch.cat([tokens, pad_tokens], dim=0)

    # Create an attention mask: 1.0 for real tokens, 0.0 for padded tokens
    attention_mask = torch.zeros((max_token_length,), dtype=torch.float32)
    attention_mask[:real_token_count] = 1.0

    return tokens, attention_mask


class CustomImageDataset(Dataset):
    """
    Custom dataset that:
      - Recursively scans a given directory for images.
      - Applies a transformation pipeline that includes resizing (with aspect ratio preservation),
        optional horizontal flip augmentation, and conversion to tensor.
      - Depending on the configuration flag, either returns the raw transformed image or 
        pre-processes the image through a pretrained VAE (encoding) and converts it into 
        a sequence of patch tokens (with padding) using patchify_and_pad.
    """
    def __init__(
        self,
        root_dir: str,
        transform: transforms.Compose,
        preprocess_latents: bool = False,
        vae: Optional[nn.Module] = None,
        patch_size: int = 2,
        max_token_length: int = 256,
    ) -> None:
        """
        Args:
            root_dir (str): Directory where images are stored.
            transform (transforms.Compose): Transformation pipeline to apply.
            preprocess_latents (bool): If True, perform VAE encoding and patch tokenization.
            vae (Optional[nn.Module]): Pretrained VAE model for latent preprocessing.
            patch_size (int): Size of the patch for tokenization.
            max_token_length (int): Maximum token sequence length.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.preprocess_latents = preprocess_latents
        self.vae = vae  # Expected to be a pretrained VAE if preprocess_latents is True.
        self.patch_size = patch_size
        self.max_token_length = max_token_length

        # Recursively collect image file paths from the root_dir.
        self.image_paths: List[str] = []
        valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")
        for current_root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.lower().endswith(valid_extensions):
                    self.image_paths.append(os.path.join(current_root, file))

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in directory {self.root_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Loads an image, applies the transformation pipeline, and returns:
          - In default mode: the transformed image and original dimensions.
          - In latent preprocessing mode: the patchified latent tokens and attention mask.
        
        Args:
            idx (int): Index of the image.

        Returns:
            Dict[str, Any]: Dictionary containing either:
                - {"image": tensor, "original_size": (width, height)} for raw images.
                OR
                - {"tokens": tensor, "attention_mask": tensor, "latent_shape": tensor.shape}.
        """
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        # Apply transformation pipeline.
        image_transformed = self.transform(image)
        
        if not self.preprocess_latents:
            # Return raw transformed image and its original size (width, height).
            return {
                "image": image_transformed,
                "original_size": image.size,
            }
        
        # Preprocessing branch: encode image to latent space and patchify.
        if self.vae is None:
            raise RuntimeError("VAE model must be provided for latent preprocessing.")
        
        # VAE expects a batch dimension.
        image_batch = image_transformed.unsqueeze(0)  # Shape: [1, C, H, W]
        with torch.no_grad():
            latent = self.vae.encode(image_batch)
            # If the encoder returns a tuple (e.g., (latent, logvar)), take the first element.
            if isinstance(latent, (tuple, list)):
                latent = latent[0]
            # Remove batch dimension: shape becomes [C, H_latent, W_latent]
            latent = latent.squeeze(0)
        
        tokens, attention_mask = patchify_and_pad(latent, self.patch_size, self.max_token_length)
        return {
            "tokens": tokens,                # Shape: [max_token_length, token_dim]
            "attention_mask": attention_mask,  # Shape: [max_token_length]
            "latent_shape": list(latent.shape) # [C, H_latent, W_latent]
        }


class DatasetLoader:
    """
    DatasetLoader handles reading the image dataset and preparing the DataLoader.
    It extracts configuration parameters from the provided config, sets up the transformation
    pipeline (including resizing with aspect ratio preservation and horizontal flip augmentation),
    and instantiates the dataset (either returning raw images or preprocessed latent tokens).
    """
    def __init__(self, config: Dict[str, Any]) -> None:
        # Extract configuration for data and training.
        data_config = config.get("data", {})
        training_config = config.get("training", {})
        model_config = config.get("model", {})

        # Use 'data_dir' key if provided; otherwise, default to './data/imagenet'
        self.root_dir: str = data_config.get("data_dir", "./data/imagenet")
        self.resize_max_area: int = data_config.get("resize_max_area", 65536)
        self.augmentation: str = data_config.get("augmentation", "Horizontal Flip")
        # Flag to determine if latent preprocessing is performed in the loader.
        self.preprocess_latents: bool = data_config.get("preprocess_latents", False)
        self.batch_size: int = training_config.get("batch_size", 256)
        self.patch_size: int = model_config.get("patch_size", 2)
        self.max_token_length: int = model_config.get("max_token_length", 256)
        self.pretrained_vae_path: str = model_config.get("pretrained_vae", "huggingface/stabilityai/sd-vae-ft-ema")

        # Build the image transformation pipeline.
        transform_list: List[Any] = []
        # Apply resizing that preserves aspect ratio.
        transform_list.append(
            transforms.Lambda(lambda img: resize_preserve_aspect(img, self.resize_max_area))
        )
        # Apply horizontal flip augmentation if specified.
        if self.augmentation.lower() == "horizontal flip":
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
        # Convert image to tensor.
        transform_list.append(transforms.ToTensor())
        self.transform: transforms.Compose = transforms.Compose(transform_list)

        # If latent preprocessing is enabled, instantiate the pretrained VAE.
        self.vae: Optional[nn.Module] = None
        if self.preprocess_latents:
            self.vae = self.load_pretrained_vae(self.pretrained_vae_path)
            self.vae.eval()  # Ensure VAE runs in evaluation mode.

    def load_pretrained_vae(self, vae_path: str) -> nn.Module:
        """
        Loads a pretrained VAE model from the specified path. In a practical implementation,
        this should load the model from Hugging Face or a local checkpoint.
        Here, we simulate the loading by returning a DummyVAE.
        
        Args:
            vae_path (str): The path or identifier for the pretrained VAE model.

        Returns:
            nn.Module: A pretrained VAE model.
        """
        # DummyVAE simulates a pretrained VAE encoder.
        class DummyVAE(nn.Module):
            def __init__(self) -> None:
                super(DummyVAE, self).__init__()
                # A simple encoder that reduces spatial resolution by a factor of 2.
                self.encoder = nn.Sequential(
                    nn.Conv2d(3, 4, kernel_size=3, stride=2, padding=1),
                    nn.ReLU()
                )

            def encode(self, x: torch.Tensor) -> torch.Tensor:
                # x shape: [B, 3, H, W] --> returns shape: [B, 4, H//2, W//2]
                return self.encoder(x)

        # In an actual implementation, replace DummyVAE() with code to load the real VAE, e.g.,
        # from diffusers import AutoencoderKL
        # return AutoencoderKL.from_pretrained(vae_path)
        return DummyVAE()

    def load_data(self) -> DataLoader:
        """
        Creates and returns a DataLoader for the dataset.
        The dataset uses the custom transformation pipeline and optionally preprocesses latents.
        
        Returns:
            DataLoader: DataLoader instance for the dataset.
        """
        dataset = CustomImageDataset(
            root_dir=self.root_dir,
            transform=self.transform,
            preprocess_latents=self.preprocess_latents,
            vae=self.vae,
            patch_size=self.patch_size,
            max_token_length=self.max_token_length,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        return dataloader
