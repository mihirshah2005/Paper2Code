"""
config.py

This module implements the Config class that serves as the single source of truth for all experiment 
parameters. It loads configuration parameters from a YAML file (default "config.yaml"), validates the 
necessary keys, and provides getter methods to access the training settings, curriculum hyperparameters, 
reproducibility seeds, and evaluation metrics. All other modules (data loader, model, curriculum scheduler, 
trainer, evaluation, and main entry) should import this Config class to ensure consistent parameter usage.
"""

import os
import yaml
import logging
from typing import Any, Dict, List, Optional


class Config:
    """
    Config class loads and validates the experiment configuration from a YAML file 
    and provides getter methods for accessing different configuration categories.
    """

    def __init__(self, config_path: str = "config.yaml") -> None:
        """
        Initialize Config instance by loading and validating the configuration.

        Args:
            config_path (str): Path to the YAML configuration file. Defaults to "config.yaml".
        """
        self.config_path: str = config_path
        self.params: Dict[str, Any] = {}
        self._load_config()
        self._validate_config()
        self._log_config_summary()

    def _load_config(self) -> None:
        """
        Loads configuration from the YAML file specified by self.config_path.
        Raises:
            FileNotFoundError: If the YAML configuration file does not exist.
            ValueError: If there is an error parsing the YAML file.
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file '{self.config_path}' not found.")
        try:
            with open(self.config_path, "r") as file:
                self.params = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            raise ValueError(f"Error parsing configuration file '{self.config_path}': {exc}")

    def _validate_config(self) -> None:
        """
        Validates that the necessary sections and keys are present in the loaded configuration.
        Raises:
            KeyError: If any required configuration key or sub-key is missing.
            ValueError: If the configuration for a given section is not of the expected type.
        """
        # Validate top-level required keys
        required_top_keys: List[str] = ["training", "curriculum", "reproducibility", "evaluation"]
        for key in required_top_keys:
            if key not in self.params:
                raise KeyError(f"Missing required configuration section: '{key}'")

        # Validate 'training' section for required domains
        required_training_domains: List[str] = ["cv", "nlp", "graph"]
        training_config = self.params["training"]
        for domain in required_training_domains:
            if domain not in training_config:
                raise KeyError(f"Missing training configuration for domain: '{domain}'")

        # For NLP training, ensure that both 'lstm' and 'transformer' configurations exist
        nlp_config = training_config.get("nlp")
        if not isinstance(nlp_config, dict):
            raise ValueError("The 'nlp' training configuration must be a dictionary with keys 'lstm' and 'transformer'.")
        for subdomain in ["lstm", "transformer"]:
            if subdomain not in nlp_config:
                raise KeyError(f"Missing NLP training configuration for subdomain: '{subdomain}'")

    def _log_config_summary(self) -> None:
        """
        Logs a summary of the loaded configuration for verification and reproducibility.
        """
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        logger.info("Configuration loaded successfully.")
        logger.info("Training Configuration:")
        logger.info(self.params.get("training", {}))
        logger.info("Curriculum Configuration:")
        logger.info(self.params.get("curriculum", {}))
        logger.info("Reproducibility Configuration:")
        logger.info(self.params.get("reproducibility", {}))
        logger.info("Evaluation Configuration:")
        logger.info(self.params.get("evaluation", {}))

    def get_training_config(self, domain: str, subdomain: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieves the training configuration for a specified domain.

        Args:
            domain (str): The domain for which to retrieve training configuration 
                          (e.g., "cv", "nlp", "graph").
            subdomain (Optional[str]): For the "nlp" domain, the subdomain can be "lstm" or "transformer".

        Returns:
            Dict[str, Any]: The training configuration dictionary for the specified domain.
        
        Raises:
            KeyError: If the configuration for the given domain or subdomain is not found.
        """
        training_config = self.params.get("training", {})
        if domain not in training_config:
            raise KeyError(f"Training configuration for domain '{domain}' not found.")
        
        if domain == "nlp" and subdomain:
            nlp_config = training_config.get("nlp", {})
            if subdomain not in nlp_config:
                raise KeyError(f"Training configuration for NLP subdomain '{subdomain}' not found.")
            return nlp_config[subdomain]
        else:
            return training_config[domain]

    def get_curriculum_config(self) -> Dict[str, Any]:
        """
        Retrieves the curriculum learning configuration.

        Returns:
            Dict[str, Any]: The curriculum configuration dictionary containing hyperparameters like
                            warmup_epochs, schedule_epochs, and growth_rate.
        """
        return self.params.get("curriculum", {})

    def get_reproducibility_config(self) -> Dict[str, Any]:
        """
        Retrieves the reproducibility configuration, including the list of random seeds.

        Returns:
            Dict[str, Any]: The reproducibility configuration dictionary.
        """
        return self.params.get("reproducibility", {})

    def get_evaluation_config(self) -> Dict[str, Any]:
        """
        Retrieves the evaluation configuration, including performance metrics and 
        flags for recording complexity (e.g., training_time and gpu_memory).

        Returns:
            Dict[str, Any]: The evaluation configuration dictionary.
        """
        return self.params.get("evaluation", {})
