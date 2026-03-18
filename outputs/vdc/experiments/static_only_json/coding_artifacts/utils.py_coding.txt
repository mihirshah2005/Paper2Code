"""utils.py

This module provides shared helper functions for configuration loading, 
random seed setup, image preprocessing, backdoor trigger injection,
logging setup, and API response caching. These functions support reproducibility,
data augmentation, and efficient API call management across the VDC project.
"""

import os
import random
import logging
import yaml
from typing import Any, Dict, Optional

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw


# ------------------------------------------------------------------------------
# 1. Configuration Loading
# ------------------------------------------------------------------------------

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Loads the experiment configuration from the YAML file.

    Args:
        config_path (str): Path to the YAML configuration file. Default "config.yaml".

    Returns:
        Dict[str, Any]: A dictionary of configuration parameters.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file '{config_path}' not found.")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


# ------------------------------------------------------------------------------
# 2. Random Seed Setup
# ------------------------------------------------------------------------------

def set_random_seed(seed: int) -> None:
    """
    Sets the random seed for Python's random, NumPy, and Torch (CPU and GPU)
    to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------------------
# 3. Image Preprocessing
# ------------------------------------------------------------------------------

def preprocess_image(image: Image.Image, 
                     target_size: tuple = (224, 224),
                     normalization: Optional[Dict[str, Any]] = None) -> torch.Tensor:
    """
    Preprocesses a PIL image: resize, convert to tensor, and apply normalization.

    Args:
        image (PIL.Image.Image): Input PIL image.
        target_size (tuple): Desired output size (width, height). Default (224, 224).
        normalization (Optional[Dict[str, Any]]): Dictionary with "mean" and "std" keys.
            Default is based on ImageNet statistics.

    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    if normalization is None:
        normalization = {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)}
    
    transform_pipeline = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(normalization["mean"]),
                             std=list(normalization["std"]))
    ])
    return transform_pipeline(image)


# ------------------------------------------------------------------------------
# 4. Trigger Overlay Functions (Backdoor Attack Implementations)
# ------------------------------------------------------------------------------

def apply_badnets_trigger(image: Image.Image, dataset: str) -> Image.Image:
    """
    Applies the BadNets backdoor trigger to the input image.
    For CIFAR-10: inserts a 3x3 white square in the lower-right corner.
    For ImageNet-100 and ImageNet-Dog: uses a 21x21 white square.

    Args:
        image (PIL.Image.Image): Input image.
        dataset (str): Dataset name ("CIFAR-10", "ImageNet-100", "ImageNet-Dog").

    Returns:
        PIL.Image.Image: Image with BadNets trigger applied.
    """
    patch_size: int = 3 if dataset == "CIFAR-10" else 21
    width, height = image.size
    # Create a white square patch
    white_patch = Image.new("RGB", (patch_size, patch_size), color=(255, 255, 255))
    # Paste the white patch at the lower-right corner
    image_with_trigger = image.copy()
    image_with_trigger.paste(white_patch, (width - patch_size, height - patch_size))
    return image_with_trigger


def apply_blended_trigger(image: Image.Image, trigger_image: Image.Image, blend_ratio: float = 0.1) -> Image.Image:
    """
    Applies a blended backdoor trigger by blending a trigger image (e.g., "Hello Kitty")
    with the original image using the specified blend ratio.

    Args:
        image (PIL.Image.Image): Input image.
        trigger_image (PIL.Image.Image): Trigger image to overlay.
        blend_ratio (float): Blending ratio for the trigger (default 0.1).

    Returns:
        PIL.Image.Image: Blended image.
    """
    # Resize the trigger to match the image size for blending.
    trigger_resized = trigger_image.resize(image.size)
    blended_image = Image.blend(image, trigger_resized, blend_ratio)
    return blended_image


def apply_sig_trigger(image: Image.Image, delta: int = 20, frequency: int = 6) -> Image.Image:
    """
    Applies a sinusoidal (SIG) backdoor trigger to the image.
    Generates a sinusoidal pattern across the image width and adds it to each channel.

    Args:
        image (PIL.Image.Image): Input image.
        delta (int): Amplitude of the sinusoidal signal. Default value 20.
        frequency (int): Frequency of the sinusoidal pattern. Default value 6.

    Returns:
        PIL.Image.Image: Image with the sinusoidal trigger applied.
    """
    np_image = np.array(image).astype(np.float32)
    height, width, channels = np_image.shape
    # Generate horizontal sinusoidal signal for each column.
    x = np.arange(width)
    pattern = delta * np.sin(2 * np.pi * x * frequency / width)
    # Tile the pattern to match the image height.
    pattern = np.tile(pattern, (height, 1))
    # Expand pattern dimensions and repeat for each channel.
    pattern = np.expand_dims(pattern, axis=2)
    pattern = np.repeat(pattern, channels, axis=2)
    # Add the sinusoidal pattern to the image and clip values.
    triggered_np = np_image + pattern
    triggered_np = np.clip(triggered_np, 0, 255).astype(np.uint8)
    return Image.fromarray(triggered_np)


def apply_trojan_trigger(image: Image.Image) -> Image.Image:
    """
    Simulates the TrojanNN backdoor trigger by overlaying a predefined pattern.
    For simplicity, draws a red rectangle at the top-left corner to indicate the trigger.

    Args:
        image (PIL.Image.Image): Input image.

    Returns:
        PIL.Image.Image: Image with the simulated TrojanNN trigger.
    """
    image_copy = image.copy()
    draw = ImageDraw.Draw(image_copy)
    # Draw a red rectangle (20x20) at the top-left corner.
    draw.rectangle([0, 0, 20, 20], fill=(255, 0, 0))
    return image_copy


def apply_ssba_trigger(image: Image.Image) -> Image.Image:
    """
    Simulates the SSBA backdoor trigger by adding subtle Gaussian noise to the image.

    Args:
        image (PIL.Image.Image): Input image.

    Returns:
        PIL.Image.Image: Image with SSBA-like trigger applied.
    """
    np_image = np.array(image).astype(np.float32)
    noise = np.random.normal(loc=0, scale=5, size=np_image.shape).astype(np.float32)
    triggered_np = np_image + noise
    triggered_np = np.clip(triggered_np, 0, 255).astype(np.uint8)
    return Image.fromarray(triggered_np)


def apply_wanet_trigger(image: Image.Image) -> Image.Image:
    """
    Simulates the WaNet backdoor trigger by applying a slight affine warp.
    For simplicity, performs a small translation.

    Args:
        image (PIL.Image.Image): Input image.

    Returns:
        PIL.Image.Image: Warped image.
    """
    width, height = image.size
    # Apply an affine transformation with a small translation (5 pixels right and down)
    transformed_image = image.transform(
        image.size,
        Image.AFFINE,
        (1, 0, 5, 0, 1, 5)
    )
    return transformed_image


# ------------------------------------------------------------------------------
# 5. Logging Facilities
# ------------------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    """
    Sets up and returns a logger for consistent logging across modules.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


# ------------------------------------------------------------------------------
# 6. API Response Caching
# ------------------------------------------------------------------------------

# Module-level in-memory cache for API responses.
_API_CACHE: Dict[str, Any] = {}

def cache_api_response(key: str, response: Any) -> None:
    """
    Caches the API response using a key.

    Args:
        key (str): Unique key to identify the API request.
        response (Any): The API response to cache.
    """
    _API_CACHE[key] = response


def get_cached_api_response(key: str) -> Optional[Any]:
    """
    Retrieves the cached API response for a given key, if available.

    Args:
        key (str): The key identifying the API response.

    Returns:
        Optional[Any]: The cached response or None if not found.
    """
    return _API_CACHE.get(key, None)
    
# ------------------------------------------------------------------------------
# End of utils.py
# ------------------------------------------------------------------------------
