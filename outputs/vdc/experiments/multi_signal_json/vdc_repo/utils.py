"""
utils.py

This module provides a collection of utility functions for the VDC project.
It includes functions for:
  - Loading and validating configuration from a YAML file (load_config)
  - Setting random seeds for reproducibility (set_random_seed)
  - Image preprocessing helpers and trigger overlay functions for backdoor attacks,
    including BadNets, Blended, SIG, TrojanNN, SSBA, and WaNet triggers.
  - API response caching with thread safety.
  - Helper functions for text cleaning and general logging.

All functions use strong typing, default values, and explicit error-checking.
"""

import os
import logging
import random
import numpy as np
import yaml
import threading
import hashlib
import json
import string
import re
from typing import Any, Dict, Optional, List, Tuple

from PIL import Image, ImageDraw

# Global in-memory cache for API responses with thread safety.
_api_cache: Dict[str, str] = {}
_cache_lock: threading.Lock = threading.Lock()


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Initializes and returns a logger with a specific format.
    
    Args:
        name (Optional[str]): Name of the logger. Defaults to None.
    
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        # Set log level from environment variable or default to INFO.
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


LOGGER = get_logger("utils")


def load_config(filepath: str) -> Dict[str, Any]:
    """
    Loads and validates the YAML configuration file.
    Mandatory keys are: "training", "detection", "dirty_sample_generation", "models", "api".
    Missing keys are filled with default values and a warning is logged.
    
    Args:
        filepath (str): Path to the YAML configuration file.
    
    Returns:
        Dict[str, Any]: Parsed and validated configuration dictionary.
    
    Raises:
        FileNotFoundError: If the configuration file does not exist.
        yaml.YAMLError: If the YAML file cannot be parsed.
    """
    if not os.path.exists(filepath):
        LOGGER.error(f"Configuration file '{filepath}' not found.")
        raise FileNotFoundError(f"Configuration file '{filepath}' not found.")

    try:
        with open(filepath, "r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        LOGGER.error(f"Error parsing YAML file: {exc}")
        raise exc

    # Validate and set default values for mandatory keys.
    defaults = {
        "training": {
            "epochs": 100,
            "optimizer": "SGD",
            "learning_rate": 0.1,
            "batch_size": {"CIFAR-10": 128, "ImageNet-100": 64, "ImageNet-Dog": 64},
            "lr_decay_schedule": {
                "CIFAR-10": [{"epoch": 50}, {"epoch": 75}],
                "ImageNet-100": [{"epoch": 30}, {"epoch": 60}],
                "ImageNet-Dog": [{"epoch": 30}, {"epoch": 60}]
            }
        },
        "detection": {
            "threshold": 0.5,
            "general_questions_count": 2,
            "label_specific_questions": {"CIFAR-10": 4, "ImageNet-100": 6, "ImageNet-Dog": 4}
        },
        "dirty_sample_generation": {
            "noisy_ratio": 0.4,
            "poisoning": {
                "CIFAR-10": {"samples_per_class": [50, 500]},
                "ImageNet-100": {"samples_per_class": [5, 50]},
                "ImageNet-Dog": {"samples_per_class": 80}
            }
        },
        "models": {
            "detector": "VDC",
            "classifier": "ResNet-18"
        },
        "api": {
            "chatgpt": {"model": "gpt-3.5-turbo"},
            "instruct_blip": {"model": "Instruct-BLIP"}
        },
        # Optional trigger configuration defaults.
        "trigger": {
            "badnets_size": {"CIFAR-10": (3, 3), "default": (21, 21)},
            "blended_image_path": "hello_kitty.png",
            "blend_ratio": 0.1,
            "sig": {"delta": 20, "frequency": 6},
            "trojan_mask_path": "apple_logo.png",
            "trojan_scale": 1.0,
            "trojan_position": None,  # None implies default to lower-right corner.
            "ssba": {"amplitude": 5},
            "wanet": {"max_translation": 3}
        }
    }

    # Check mandatory keys and set missing ones with defaults.
    for key, default_value in defaults.items():
        if key not in config:
            LOGGER.warning(f"Key '{key}' missing in configuration. Using default: {default_value}")
            config[key] = default_value
        else:
            # For nested dictionaries, verify individual keys.
            if isinstance(default_value, dict):
                for subkey, subdefault in default_value.items():
                    if subkey not in config[key]:
                        LOGGER.warning(f"Key '{key}.{subkey}' missing in configuration. Using default: {subdefault}")
                        config[key][subkey] = subdefault

    return config


def set_random_seed(seed: int) -> None:
    """
    Sets the random seed for Python, NumPy, and PyTorch (including CUDA) for reproducibility.
    
    Args:
        seed (int): The seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior in cuDNN.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        LOGGER.warning("torch not available; skipping torch seed setup.")
    LOGGER.info(f"Random seed set to {seed}.")


def get_config_value(config: Dict[str, Any], key: str, default: Any) -> Any:
    """
    Retrieves a nested configuration value using a dot-separated key.
    
    Args:
        config (Dict[str, Any]): The configuration dictionary.
        key (str): Dot-separated key string (e.g., "trigger.sig.delta").
        default (Any): Default value if key not found.
    
    Returns:
        Any: The retrieved configuration value or the default.
    """
    keys = key.split('.')
    value = config
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            LOGGER.warning(f"Configuration key '{key}' not found. Using default: {default}")
            return default
    return value


def inject_badnets_trigger(image: Any, dataset: str = "CIFAR-10", config: Optional[Dict[str, Any]] = None) -> Any:
    """
    Applies the BadNets backdoor trigger to an image by overlaying a white square.
    The trigger size and default position (lower-right corner) are determined by dataset type.
    
    Args:
        image (Any): Input image (expected as a PIL.Image.Image or convertible to one).
        dataset (str): Identifier for dataset type ("CIFAR-10", "ImageNet-100", etc.).
        config (Optional[Dict[str, Any]]): Configuration dictionary. If None, defaults are used.
    
    Returns:
        Any: The image with the injected trigger.
    """
    if not isinstance(image, Image.Image):
        try:
            image = Image.fromarray(np.uint8(image))
        except Exception as e:
            LOGGER.error(f"Could not convert image to PIL format: {e}")
            return image

    width, height = image.size
    # Get trigger size from config; use default values.
    trigger_sizes: Dict[str, Tuple[int, int]] = get_config_value(
        config if config else {}, "trigger.badnets_size", {"CIFAR-10": (3, 3), "default": (21, 21)}
    )
    if dataset in trigger_sizes:
        trigger_size = trigger_sizes[dataset]
    else:
        trigger_size = trigger_sizes.get("default", (21, 21))

    # Ensure trigger size fits in the image
    t_width, t_height = trigger_size
    if t_width > width or t_height > height:
        LOGGER.warning(f"Trigger size {trigger_size} exceeds image dimensions {image.size}. Adjusting trigger size.")
        t_width = min(t_width, width)
        t_height = min(t_height, height)
        trigger_size = (t_width, t_height)

    # Default position: lower-right corner.
    position = (width - t_width, height - t_height)
    trigger_square = Image.new("RGB", trigger_size, color=(255, 255, 255))
    # Create a copy to avoid modifying the original image directly.
    new_image = image.copy()
    new_image.paste(trigger_square, position)
    return new_image


def apply_blended_trigger(image: Any, config: Optional[Dict[str, Any]] = None) -> Any:
    """
    Applies the Blended trigger by blending a trigger image onto the original image.
    Parameters such as the trigger image path and blend ratio are retrieved from the configuration.
    
    Args:
        image (Any): Input image (PIL.Image.Image preferred).
        config (Optional[Dict[str, Any]]): Configuration dictionary.
    
    Returns:
        Any: The image after blending with the trigger.
    """
    if not isinstance(image, Image.Image):
        try:
            image = Image.fromarray(np.uint8(image))
        except Exception as e:
            LOGGER.error(f"Could not convert image to PIL format: {e}")
            return image

    # Retrieve parameters from config with defaults.
    trigger_image_path = get_config_value(
        config if config else {}, "trigger.blended_image_path", "hello_kitty.png"
    )
    blend_ratio: float = get_config_value(
        config if config else {}, "trigger.blend_ratio", 0.1
    )
    if not os.path.exists(trigger_image_path):
        LOGGER.error(f"Blended trigger image file '{trigger_image_path}' not found. Returning original image.")
        return image

    try:
        trigger_img = Image.open(trigger_image_path).convert("RGB")
    except Exception as e:
        LOGGER.error(f"Error loading blended trigger image: {e}. Returning original image.")
        return image

    # Resize trigger image to match original image dimensions.
    trigger_img = trigger_img.resize(image.size)
    try:
        blended_image = Image.blend(image, trigger_img, blend_ratio)
    except Exception as e:
        LOGGER.error(f"Error blending images: {e}. Returning original image.")
        return image
    return blended_image


def apply_sig_trigger(image: Any, config: Optional[Dict[str, Any]] = None) -> Any:
    """
    Applies the SIG trigger by adding a horizontal sinusoidal noise to the image.
    The parameters delta and frequency are retrieved from the configuration.
    
    Args:
        image (Any): Input image (expected as a PIL.Image.Image).
        config (Optional[Dict[str, Any]]): Configuration dictionary.
    
    Returns:
        Any: The image with the added sinusoidal noise trigger.
    """
    if not isinstance(image, Image.Image):
        try:
            image = Image.fromarray(np.uint8(image))
        except Exception as e:
            LOGGER.error(f"Could not convert image to PIL format: {e}")
            return image

    delta: float = get_config_value(config if config else {}, "trigger.sig.delta", 20)
    frequency: float = get_config_value(config if config else {}, "trigger.sig.frequency", 6)

    image_array = np.array(image).astype(np.float32)
    if image_array.ndim != 3:
        LOGGER.error("SIG trigger expected a 3-channel image. Returning original image.")
        return image

    h, w, c = image_array.shape
    # Create a horizontal sinusoidal signal.
    x = np.arange(w)
    sine_wave = delta * np.sin(2 * np.pi * frequency * x / w)
    # Broadcast to full image shape.
    noise = np.tile(sine_wave, (h, 1))
    noise = np.expand_dims(noise, axis=2)
    noise = np.repeat(noise, c, axis=2)
    # Add noise and clip. Assume image pixel range is 0-255.
    noisy_image = image_array + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_image)


def apply_trojan_trigger(image: Any, config: Optional[Dict[str, Any]] = None) -> Any:
    """
    Applies the TrojanNN trigger by overlaying a trigger mask (e.g., Apple logo) onto the image.
    Placement and scaling parameters are retrieved from the configuration.
    
    Args:
        image (Any): Input image (PIL.Image.Image preferred).
        config (Optional[Dict[str, Any]]): Configuration dictionary.
    
    Returns:
        Any: The image with the trojan trigger overlaid.
    """
    if not isinstance(image, Image.Image):
        try:
            image = Image.fromarray(np.uint8(image))
        except Exception as e:
            LOGGER.error(f"Could not convert image to PIL format: {e}")
            return image

    trigger_mask_path = get_config_value(config if config else {}, "trigger.trojan_mask_path", "apple_logo.png")
    trojan_scale: float = get_config_value(config if config else {}, "trigger.trojan_scale", 1.0)
    trojan_position = get_config_value(config if config else {}, "trigger.trojan_position", None)

    if not os.path.exists(trigger_mask_path):
        LOGGER.error(f"Trojan mask file '{trigger_mask_path}' not found. Returning original image.")
        return image

    try:
        mask = Image.open(trigger_mask_path).convert("RGBA")
    except Exception as e:
        LOGGER.error(f"Error loading trojan mask image: {e}. Returning original image.")
        return image

    # Resize the mask according to the scale.
    mask_width, mask_height = mask.size
    new_size = (int(mask_width * trojan_scale), int(mask_height * trojan_scale))
    mask = mask.resize(new_size, resample=Image.ANTIALIAS)

    width, height = image.size
    # Default position: lower-right corner.
    if trojan_position is None:
        position = (width - new_size[0], height - new_size[1])
    else:
        # Ensure provided position is a tuple of two integers.
        if (isinstance(trojan_position, (list, tuple)) and len(trojan_position) == 2):
            position = (int(trojan_position[0]), int(trojan_position[1]))
        else:
            LOGGER.warning(f"Invalid trojan_position provided. Using default lower-right corner.")
            position = (width - new_size[0], height - new_size[1])
    # Overlay the mask using alpha composite.
    try:
        image_rgba = image.convert("RGBA")
        image_rgba.paste(mask, position, mask)
        return image_rgba.convert("RGB")
    except Exception as e:
        LOGGER.error(f"Error applying Trojan trigger: {e}. Returning original image.")
        return image


def apply_ssba_trigger(image: Any, config: Optional[Dict[str, Any]] = None) -> Any:
    """
    Simulates the SSBA trigger by adding low-amplitude Gaussian noise to the image.
    
    Args:
        image (Any): Input image (PIL.Image.Image preferred).
        config (Optional[Dict[str, Any]]): Configuration dictionary.
    
    Returns:
        Any: The image with simulated SSBA noise added.
    """
    if not isinstance(image, Image.Image):
        try:
            image = Image.fromarray(np.uint8(image))
        except Exception as e:
            LOGGER.error(f"Could not convert image to PIL format: {e}")
            return image

    amplitude: float = get_config_value(config if config else {}, "trigger.ssba.amplitude", 5)
    image_array = np.array(image).astype(np.float32)
    noise = np.random.normal(loc=0.0, scale=amplitude, size=image_array.shape)
    noisy_image = image_array + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_image)


def apply_wanet_trigger(image: Any, config: Optional[Dict[str, Any]] = None) -> Any:
    """
    Simulates the WaNet trigger by applying a slight affine translation to the image.
    This is a simplified version of elastic warping.
    
    Args:
        image (Any): Input image (PIL.Image.Image preferred).
        config (Optional[Dict[str, Any]]): Configuration dictionary.
    
    Returns:
        Any: The image with a slight affine transformation applied.
    """
    if not isinstance(image, Image.Image):
        try:
            image = Image.fromarray(np.uint8(image))
        except Exception as e:
            LOGGER.error(f"Could not convert image to PIL format: {e}")
            return image

    max_translation: int = get_config_value(config if config else {}, "trigger.wanet.max_translation", 3)
    width, height = image.size
    # Generate a small random translation (deterministic by using a fixed seed here)
    rng = random.Random(42)
    dx = rng.randint(-max_translation, max_translation)
    dy = rng.randint(-max_translation, max_translation)
    # Define an affine transformation matrix for translation.
    affine_matrix = (1, 0, dx, 0, 1, dy)
    try:
        warped_image = image.transform(image.size, Image.AFFINE, affine_matrix, resample=Image.BILINEAR)
        return warped_image
    except Exception as e:
        LOGGER.error(f"Error applying WaNet trigger: {e}. Returning original image.")
        return image


def generate_cache_key(*args, **kwargs) -> str:
    """
    Generates a deterministic cache key based on the provided arguments.
    
    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.
    
    Returns:
        str: A SHA256 hash string representing the cache key.
    """
    try:
        key_data = {
            "args": args,
            "kwargs": kwargs
        }
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.sha256(key_string.encode('utf-8')).hexdigest()
        return key_hash
    except Exception as e:
        LOGGER.error(f"Error generating cache key: {e}")
        return ""


def get_cached_api_response(key: str) -> Optional[str]:
    """
    Retrieves an API response from the cache if it exists.
    
    Args:
        key (str): The cache key.
    
    Returns:
        Optional[str]: Cached response string if found, else None.
    """
    with _cache_lock:
        return _api_cache.get(key, None)


def set_cached_api_response(key: str, response: str) -> None:
    """
    Stores an API response in the cache.
    
    Args:
        key (str): The cache key.
        response (str): The API response.
    """
    with _cache_lock:
        _api_cache[key] = response


def clean_text(text: str) -> str:
    """
    Cleans text by converting to lowercase, removing punctuation, and stripping extra whitespace.
    
    Args:
        text (str): The input text.
    
    Returns:
        str: The cleaned text.
    """
    text_lower = text.lower()
    # Remove punctuation using regex.
    text_no_punct = re.sub(f"[{re.escape(string.punctuation)}]", "", text_lower)
    # Remove extra whitespace.
    cleaned = re.sub(r"\s+", " ", text_no_punct).strip()
    return cleaned
