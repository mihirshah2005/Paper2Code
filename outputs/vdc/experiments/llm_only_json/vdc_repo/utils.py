"""
utils.py

This module provides common utility functions for the VDC system.
It handles configuration loading, seed initialization, image preprocessing,
trigger overlay functions for backdoor attacks, API caching and logging,
and text cleaning/similarity utilities.

Dependencies:
  - numpy==1.21.0
  - torch==1.9.0
  - torchvision==0.10.0
  - transformers==4.15.0
  - openai==0.27.0
  - requests==2.26.0
  - PyYAML (for YAML configuration parsing)

Author: Your Name
"""

import os
import random
import logging
import re
import hashlib
from typing import Any, Dict, Optional, Union
import yaml

import numpy as np
import torch
from PIL import Image, ImageDraw

from torchvision import transforms

# Global API cache dictionary
API_CACHE: Dict[str, str] = {}


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Loads the YAML configuration file and returns a dictionary of settings.
    
    Args:
        config_path: Path to the YAML configuration file.
    
    Returns:
        A dictionary containing configuration parameters.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def init_seed(seed: int = 42) -> None:
    """
    Initializes random seeds for Python, NumPy, and Torch to ensure reproducibility.
    
    Args:
        seed: The seed value to set (default 42).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for GPU operations.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def preprocess_image(image: Image.Image, dataset_name: str, config: Optional[Dict[str, Any]] = None) -> torch.Tensor:
    """
    Preprocesses a PIL image into a Torch tensor by applying resizing, normalization,
    and conversion to tensor. Normalization parameters are chosen based on the dataset.
    
    Args:
        image: Input PIL Image.
        dataset_name: Name of the dataset ("CIFAR-10", "ImageNet-100", or "ImageNet-Dog").
        config: Optional configuration dictionary to override defaults.
    
    Returns:
        A Torch tensor representing the preprocessed image.
    """
    if dataset_name == "CIFAR-10":
        # Defaults for CIFAR-10
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        target_size = (32, 32)
    elif dataset_name in ["ImageNet-100", "ImageNet-Dog"]:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        target_size = (224, 224)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    preprocess = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return preprocess(image)


def tensor_to_image(tensor: torch.Tensor, dataset_name: str) -> Image.Image:
    """
    Converts a Torch tensor back into a PIL Image for visualization.
    Applies de-normalization based on the dataset settings.
    
    Args:
        tensor: A Torch tensor representing the image.
        dataset_name: Name of the dataset ("CIFAR-10", "ImageNet-100", or "ImageNet-Dog").
    
    Returns:
        A PIL Image.
    """
    if dataset_name == "CIFAR-10":
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2023, 0.1994, 0.2010])
    elif dataset_name in ["ImageNet-100", "ImageNet-Dog"]:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    image_np = tensor.cpu().numpy().transpose(1, 2, 0)
    image_np = (image_np * std + mean) * 255.0
    image_np = np.clip(image_np, 0, 255).astype(np.uint8)
    return Image.fromarray(image_np)


def overlay_badnets_trigger(image: Image.Image, dataset_name: str, config: Optional[Dict[str, Any]] = None) -> Image.Image:
    """
    Overlays a BadNets trigger (a white square) onto the input image.
    
    For CIFAR-10, a 3x3 white square is drawn in the lower-right corner.
    For ImageNet-based datasets, a 21x21 white square is used.
    
    Args:
        image: A PIL Image.
        dataset_name: Name of the dataset.
        config: Optional configuration dictionary. If provided, can override the default trigger size.
    
    Returns:
        A modified PIL image with the trigger overlaid.
    """
    # Determine trigger size; check config if available.
    default_size = 3 if dataset_name == "CIFAR-10" else 21
    trigger_size = default_size
    if config is not None:
        ds_config = config.get("dirty_sample_generation", {})
        trigger_size = ds_config.get("badnets_trigger_size", default_size)
    
    # Ensure image is in RGB mode.
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    width, height = image.size
    draw = ImageDraw.Draw(image)
    # Calculate coordinates for the trigger: bottom-right corner.
    x0 = width - trigger_size
    y0 = height - trigger_size
    x1 = width
    y1 = height
    draw.rectangle([x0, y0, x1, y1], fill=(255, 255, 255))
    return image


def overlay_blended_trigger(image: Image.Image, trigger_image: Image.Image, blend_ratio: float = 0.1) -> Image.Image:
    """
    Blends a trigger image (e.g., "Hello Kitty") with the original image using the specified blend ratio.
    
    Args:
        image: The original PIL Image.
        trigger_image: The trigger PIL Image to blend with the original.
        blend_ratio: The blending ratio (default 0.1).
    
    Returns:
        A blended PIL Image.
    """
    # Resize trigger_image to match the original image's size.
    trigger_resized = trigger_image.resize(image.size)
    # Blend the images.
    blended = Image.blend(image, trigger_resized, alpha=blend_ratio)
    return blended


def apply_sig_trigger(image: Image.Image, delta: int = 20, frequency: int = 6) -> Image.Image:
    """
    Applies a SIG trigger by overlaying a horizontal sinusoidal signal on the image.
    
    The signal is added to each pixel value based on the column index:
        v(i, j) = delta * sin(2π * j * frequency / m),
    where m is the width of the image.
    
    Args:
        image: A PIL Image.
        delta: The amplitude of the sinusoidal signal (default 20).
        frequency: The frequency of the sinusoidal signal (default 6).
    
    Returns:
        The modified PIL Image with the SIG trigger applied.
    """
    # Convert image to numpy array.
    image_np = np.array(image).astype(np.float32)
    height, width, channels = image_np.shape
    # Create a sinusoidal signal for each column.
    j_indices = np.arange(width).reshape(1, width)
    signal = delta * np.sin(2 * np.pi * j_indices * frequency / width)
    # Broadcast signal across rows and channels.
    signal = np.repeat(signal, height, axis=0)
    signal = np.expand_dims(signal, axis=2)  # shape becomes (height, width, 1)
    image_np += signal  # Add the signal to every channel.
    image_np = np.clip(image_np, 0, 255).astype(np.uint8)
    return Image.fromarray(image_np)


def apply_trojanNN_trigger(image: Image.Image, trigger_mask: Image.Image) -> Image.Image:
    """
    Applies a TrojanNN trigger by overlaying a predefined trigger mask (e.g., an Apple logo)
    onto the input image. The trigger mask is pasted onto the lower-right corner of the image.
    If the trigger mask has an alpha channel, it is used for transparency.
    
    Args:
        image: A PIL Image.
        trigger_mask: A PIL Image representing the trigger mask.
    
    Returns:
        The modified PIL image with the TrojanNN trigger applied.
    """
    # Ensure both images are in RGBA mode for proper blending.
    image_rgba = image.convert("RGBA")
    mask_rgba = trigger_mask.convert("RGBA")
    # Resize trigger mask if necessary; Here we assume the trigger mask size is defined by its own dimensions.
    mask_width, mask_height = mask_rgba.size
    img_width, img_height = image_rgba.size
    # Position: bottom-right corner.
    x = img_width - mask_width
    y = img_height - mask_height
    # Paste the mask onto the original image using its alpha channel.
    image_rgba.paste(mask_rgba, (x, y), mask_rgba)
    return image_rgba.convert("RGB")


def apply_ssba_trigger(image: Image.Image, ssba_params: Dict[str, Any]) -> Image.Image:
    """
    Applies a simplified SSBA-style trigger injection by adding imperceptible noise to the image.
    
    The ssba_params dictionary should contain key "noise_intensity" (default 10).
    
    Args:
        image: A PIL Image.
        ssba_params: Dictionary containing SSBA parameters.
    
    Returns:
        The modified PIL Image with SSBA trigger applied.
    """
    noise_intensity = ssba_params.get("noise_intensity", 10)
    # Convert image to numpy float array.
    image_np = np.array(image).astype(np.float32)
    # Generate Gaussian noise.
    noise = np.random.normal(0, noise_intensity, image_np.shape)
    image_np += noise
    image_np = np.clip(image_np, 0, 255).astype(np.uint8)
    return Image.fromarray(image_np)


def apply_wanet_trigger(image: Image.Image, warping_params: Dict[str, Any]) -> Image.Image:
    """
    Applies a simplified WaNet trigger by performing an elastic warping of the image.
    
    The warping_params dictionary should include:
        - "intensity": The intensity of the warp (default 5).
        - "scale": The scale factor for horizontal displacement (default 5).
    
    The function shifts each row horizontally by an offset computed as:
        offset = int(scale * sin(2π * row / height))
    
    Args:
        image: A PIL Image.
        warping_params: Dictionary containing warping parameters.
    
    Returns:
        The warped PIL Image.
    """
    intensity = warping_params.get("intensity", 5)
    scale = warping_params.get("scale", 5)
    image_np = np.array(image)
    height, width, channels = image_np.shape
    warped_image = np.zeros_like(image_np)
    
    # For each row, compute an offset and shift the row
    for i in range(height):
        offset = int(scale * np.sin(2 * np.pi * i / height) * intensity)
        # Use np.roll to shift the row horizontally by computed offset.
        warped_image[i] = np.roll(image_np[i], shift=offset, axis=0)
    
    return Image.fromarray(warped_image)


def generate_api_cache_key(input_params: Any) -> str:
    """
    Generates a unique cache key based on the input parameters by serializing them
    and computing a SHA256 hash.
    
    Args:
        input_params: Input parameters (can be any serializable object).
    
    Returns:
        A string representing the unique cache key.
    """
    # Convert the input parameters to a sorted string representation.
    key_str = str(input_params)
    hash_key = hashlib.sha256(key_str.encode("utf-8")).hexdigest()
    return hash_key


def cache_api_response(key: str, response: str) -> None:
    """
    Caches an API response under the given key.
    
    Args:
        key: Unique cache key.
        response: API response text.
    """
    global API_CACHE
    API_CACHE[key] = response


def get_cached_response(key: str) -> Optional[str]:
    """
    Retrieves a cached API response if it exists.
    
    Args:
        key: The cache key.
    
    Returns:
        The cached API response, or None if not found.
    """
    return API_CACHE.get(key)


def setup_logging(log_file: Optional[str] = None) -> None:
    """
    Configures logging to output messages with timestamps and log levels.
    If log_file is provided, logs are also written to that file.
    
    Args:
        log_file: Optional path to a log file.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    
    # Console handler.
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)


def clean_text(text: str) -> str:
    """
    Cleans the input text by lowercasing, removing punctuation, and extra spaces.
    
    Args:
        text: The string to clean.
    
    Returns:
        A cleaned version of the text.
    """
    text = text.lower()
    # Remove punctuation using regex.
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra whitespace.
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def simple_similarity(text1: str, text2: str) -> float:
    """
    Computes a simple Jaccard similarity index between two texts based on word tokens.
    
    Args:
        text1: First text string.
        text2: Second text string.
    
    Returns:
        A float similarity score between 0 and 1.
    """
    set1 = set(clean_text(text1).split())
    set2 = set(clean_text(text2).split())
    if not set1 or not set2:
        return 0.0
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    similarity = len(intersection) / len(union)
    return similarity
