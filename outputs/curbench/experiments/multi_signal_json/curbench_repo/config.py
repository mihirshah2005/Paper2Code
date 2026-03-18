"""config.py

This module provides the Config class which is responsible for loading,
validating, and exposing the experiment configuration as defined in the
"config.yaml" file. This configuration includes training parameters,
curriculum hyperparameters, reproducibility settings, and evaluation metrics.

The Config class ensures that:
  - The configuration file exists and is valid YAML.
  - The YAML content is a dictionary.
  - All required top-level keys ("training", "curriculum", "reproducibility", "evaluation")
    are present.
  - The reproducibility section contains a list of seeds.

Downstream modules should import this Config class and access the configuration
via its read-only 'config' property.
"""

import os
import yaml
from typing import Any, Dict


class Config:
    """Config class to load and manage experiment configuration parameters.

    The configuration is read from a YAML file (default: "config.yaml"). It is
    validated to ensure that all required keys are present and that the file
    contains a valid dictionary structure.

    Attributes:
        _config (Dict[str, Any]): Internal dictionary storing the configuration.
    """

    # Define the required top-level keys in the configuration.
    REQUIRED_KEYS = {"training", "curriculum", "reproducibility", "evaluation"}

    def __init__(self, file_path: str = "config.yaml") -> None:
        """
        Initializes the Config instance by loading and validating the YAML configuration.

        Args:
            file_path (str, optional): Path to the configuration YAML file.
                Defaults to "config.yaml".

        Raises:
            FileNotFoundError: If the configuration file is not found.
            ValueError: If YAML parsing fails or if the file content is empty or not a dictionary.
            KeyError: If any required configuration keys are missing.
        """
        self._config: Dict[str, Any] = self._load_config(file_path)
        self._validate_config(self._config)

    @property
    def config(self) -> Dict[str, Any]:
        """Provides read-only access to the loaded configuration dictionary."""
        return self._config

    def _load_config(self, file_path: str) -> Dict[str, Any]:
        """
        Loads and parses the YAML configuration file.

        Args:
            file_path (str): The file path to the YAML configuration file.

        Returns:
            Dict[str, Any]: The parsed configuration as a dictionary.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is empty or the content is not a dictionary.
            ValueError: If an error occurs during YAML parsing.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Configuration file not found at path: {file_path}")

        try:
            with open(file_path, "r") as f:
                config_data: Any = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")

        if config_data is None or not isinstance(config_data, dict):
            raise ValueError("Configuration file is empty or must contain a dictionary at the root.")

        return config_data

    def _validate_config(self, config_data: Dict[str, Any]) -> None:
        """
        Validates that the configuration dictionary contains all required top-level keys.
        Also verifies that the reproducibility section contains a 'seeds' key with a list.

        Args:
            config_data (Dict[str, Any]): The configuration dictionary to validate.

        Raises:
            KeyError: If any required top-level keys are missing.
            KeyError: If the 'seeds' key is missing under 'reproducibility'.
            ValueError: If the 'seeds' value is not a list.
        """
        missing_keys = self.REQUIRED_KEYS - set(config_data.keys())
        if missing_keys:
            raise KeyError(f"Missing required configuration keys: {', '.join(missing_keys)}")

        # Validate reproducibility settings
        reproducibility = config_data.get("reproducibility")
        if not isinstance(reproducibility, dict):
            raise ValueError("The 'reproducibility' section must be a dictionary.")
        if "seeds" not in reproducibility:
            raise KeyError("Configuration must include the 'seeds' key under 'reproducibility'.")
        if not isinstance(reproducibility["seeds"], list):
            raise ValueError("The 'seeds' in 'reproducibility' must be provided as a list of integers.")

        # Additional validations for other sections can be added here if necessary.
