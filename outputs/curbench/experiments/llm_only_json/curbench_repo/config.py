"""config.py: Singleton configuration management for CurBench project.

This module loads configuration parameters from a YAML file (default "config.yaml")
and validates the presence of required keys. It provides helper getter methods
to access different sections (training, curriculum, reproducibility, evaluation)
as deep copies, ensuring immutability. This implementation is thread-safe using
a class-level lock in the __new__ method.
"""

import os
import yaml
import threading
import copy
import logging

# Configure logging for this module.
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


class Config:
    """Singleton Config class for loading and validating configuration from a YAML file.

    The configuration file must define the following top-level keys:
        - training: Contains domain-specific training configs for 'cv', 'nlp', and 'graph'.
        - curriculum: Contains hyperparameters for curriculum learning (e.g., warmup_epochs, schedule_epochs, growth_rate).
        - reproducibility: Contains settings for reproducibility (e.g., seeds).
        - evaluation: Contains evaluation metric configuration (e.g., metrics, record_complexity).
    """

    _instance = None
    _lock = threading.Lock()  # Class-level lock for thread-safe singleton instantiation

    def __new__(cls, config_file: str = "config.yaml"):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(Config, cls).__new__(cls)
        return cls._instance

    def __init__(self, config_file: str = "config.yaml"):
        # Prevent reinitialization if already initialized.
        if hasattr(self, '_initialized') and self._initialized:
            return

        self._initialized = True  # Mark as initialized

        # Check if configuration file exists.
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file '{config_file}' not found.")

        # Load and parse the YAML configuration.
        try:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
        except yaml.YAMLError as yaml_error:
            raise yaml.YAMLError(f"Error parsing configuration file '{config_file}': {yaml_error}")

        # Validate that the loaded configuration is a dictionary.
        if not isinstance(config_data, dict):
            raise ValueError("The configuration file must contain a YAML dictionary at the top level.")

        # Validate required top-level keys.
        required_top_keys = ["training", "curriculum", "reproducibility", "evaluation"]
        for key in required_top_keys:
            if key not in config_data:
                raise KeyError(f"Required configuration key '{key}' is missing.")

        # Validate 'training' section.
        training_data = config_data.get("training")
        if not isinstance(training_data, dict):
            raise ValueError("The 'training' section must be a dictionary.")
        required_training_domains = ["cv", "nlp", "graph"]
        for domain in required_training_domains:
            if domain not in training_data:
                raise KeyError(f"Training configuration for domain '{domain}' is missing.")

        # Validate 'reproducibility' section.
        reproducibility_data = config_data.get("reproducibility", {})
        seeds = reproducibility_data.get("seeds")
        if not isinstance(seeds, list) or len(seeds) == 0:
            raise KeyError("Reproducibility configuration must include a non-empty list of 'seeds' under the 'seeds' key.")

        # Validate 'evaluation' section.
        evaluation_data = config_data.get("evaluation")
        if not isinstance(evaluation_data, dict):
            raise ValueError("The 'evaluation' section must be a dictionary.")
        if "metrics" not in evaluation_data:
            raise KeyError("Evaluation configuration must include the 'metrics' key.")

        # Optionally, you might add further validations for curriculum if needed.
        # Here we require the 'curriculum' key to be present; values can be None/default.
        curriculum_data = config_data.get("curriculum")
        if curriculum_data is None:
            raise KeyError("The 'curriculum' configuration is missing.")

        # Store configuration privately.
        self.__config = config_data
        LOGGER.info("Configuration loaded and validated successfully.")

    def get_config(self) -> dict:
        """Returns a deep copy of the entire configuration dictionary to ensure immutability."""
        return copy.deepcopy(self.__config)

    def get_training_config(self, domain: str) -> dict:
        """Returns a deep copy of the training configuration for the specified domain.

        Args:
            domain (str): The domain for which to retrieve the training configuration ('cv', 'nlp', or 'graph').

        Raises:
            KeyError: If the specified domain is not found in the training configuration.

        Returns:
            dict: A deep copy of the configuration for the specified domain.
        """
        training_config = self.__config.get("training", {})
        if domain not in training_config:
            raise KeyError(f"Training configuration for domain '{domain}' not found.")
        return copy.deepcopy(training_config[domain])

    def get_curriculum_params(self) -> dict:
        """Returns a deep copy of the curriculum learning parameters.

        Returns:
            dict: A deep copy of the 'curriculum' section of the configuration.
        """
        return copy.deepcopy(self.__config.get("curriculum", {}))

    def get_evaluation_config(self) -> dict:
        """Returns a deep copy of the evaluation configuration.

        Raises:
            KeyError: If the evaluation configuration is missing.

        Returns:
            dict: A deep copy of the 'evaluation' section.
        """
        evaluation_config = self.__config.get("evaluation", {})
        if not evaluation_config:
            raise KeyError("Evaluation configuration is missing.")
        return copy.deepcopy(evaluation_config)

    def get_reproducibility_seeds(self) -> list:
        """Returns a deep copy of the list of reproducibility seeds.

        Raises:
            KeyError: If the 'seeds' key is missing in the reproducibility section.

        Returns:
            list: A deep copy of the seeds list.
        """
        seeds = self.__config.get("reproducibility", {}).get("seeds", None)
        if seeds is None:
            raise KeyError("Reproducibility configuration must include the 'seeds' key.")
        return copy.deepcopy(seeds)
