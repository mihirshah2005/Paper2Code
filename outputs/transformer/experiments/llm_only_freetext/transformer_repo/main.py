"""main.py

This file is the entry point that orchestrates the experiment workflow.
It performs the following:
1. Loads and validates the configuration from "config.yaml".
2. Sets up logging and random seeds for reproducibility.
3. Instantiates the DatasetLoader, loads the train/validation/test DataLoaders.
4. Builds the TransformerModel based on the configuration (using the base model by default).
5. Sets up the Trainer and runs the training loop.
6. Instantiates the Evaluation module and runs beam search decoding to compute evaluation metrics.
All configuration values have explicit defaults and warnings are logged when falling back.
"""

import os
import sys
import logging
import random
import math
from typing import Any, Dict

import yaml
import torch
import numpy as np

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class Main:
    def __init__(self) -> None:
        """
        Initializes the experiment by loading and validating the configuration file,
        and setting up experiment metadata such as random seeds.
        """
        self.config = self.load_and_validate_config("config.yaml")
        self.log_effective_config()
        self.set_random_seeds(seed_value=42)

    def load_and_validate_config(self, filepath: str) -> Dict[str, Any]:
        """
        Loads the YAML configuration file and validates critical sections and keys.
        Missing optional keys trigger warnings and fallback defaults.
        Missing mandatory sections result in termination.
        
        Args:
            filepath: Path to the configuration file.
        
        Returns:
            A dictionary containing the configuration.
        """
        # Load configuration file
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        except Exception as e:
            logging.critical(f"Error reading configuration file '{filepath}': {e}")
            sys.exit(1)

        # Ensure required top-level sections exist
        required_sections = ["training", "model", "data", "inference", "checkpoint"]
        for section in required_sections:
            if section not in config:
                logging.critical(f"Configuration file missing required section: '{section}'")
                sys.exit(1)

        # Validate training section keys with defaults
        training_defaults = {
            "optimizer": "Adam",
            "beta1": 0.9,
            "beta2": 0.98,
            "epsilon": 1e-9,
            "learning_rate_schedule": "d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))",
            "warmup_steps": 4000,
            "batch_size_tokens": 25000,
            "dropout_rate": 0.1,
            "label_smoothing": 0.1
        }
        for key, default in training_defaults.items():
            if key not in config["training"]:
                logging.warning(
                    f"'training.{key}' not found in config. Falling back to default value: {default}"
                )
                config["training"][key] = default

        # 'total_steps' is a nested key
        if "total_steps" not in config["training"]:
            logging.warning(
                "'training.total_steps' not found in config. Falling back to defaults: base=100000, big=300000"
            )
            config["training"]["total_steps"] = {"base": 100000, "big": 300000}
        else:
            # Ensure both 'base' and 'big' keys exist
            if "base" not in config["training"]["total_steps"]:
                logging.warning(
                    "'training.total_steps.base' not found. Falling back to default value: 100000"
                )
                config["training"]["total_steps"]["base"] = 100000
            if "big" not in config["training"]["total_steps"]:
                logging.warning(
                    "'training.total_steps.big' not found. Falling back to default value: 300000"
                )
                config["training"]["total_steps"]["big"] = 300000

        # Validate model section: require 'base' and 'big' configurations.
        if "base" not in config["model"]:
            logging.warning(
                "'model.base' not found in config. Falling back to default base model values: "
                "num_layers=6, d_model=512, d_ff=2048, num_heads=8, d_k=64, d_v=64"
            )
            config["model"]["base"] = {
                "num_layers": 6,
                "d_model": 512,
                "d_ff": 2048,
                "num_heads": 8,
                "d_k": 64,
                "d_v": 64
            }
        if "big" not in config["model"]:
            logging.warning(
                "'model.big' not found in config. Falling back to default big model values: "
                "num_layers=6, d_model=1024, d_ff=4096, num_heads=16, d_k=64, d_v=64"
            )
            config["model"]["big"] = {
                "num_layers": 6,
                "d_model": 1024,
                "d_ff": 4096,
                "num_heads": 16,
                "d_k": 64,
                "d_v": 64
            }

        # Validate data section for translation and parsing
        if "translation" not in config["data"]:
            logging.warning(
                "'data.translation' not found in config. Falling back to defaults for translation data."
            )
            config["data"]["translation"] = {
                "datasets": {"english_german": "WMT14_en_de", "english_french": "WMT14_en_fr"},
                "vocabulary": {"english_german": 37000, "english_french": 32000}
            }
        if "parsing" not in config["data"]:
            logging.warning(
                "'data.parsing' not found in config. Falling back to defaults for parsing data."
            )
            config["data"]["parsing"] = {
                "dataset": "WSJ_Penn_Treebank",
                "vocabulary": {"wsj_only": 16000, "semi_supervised": 32000}
            }

        # Validate inference section defaults for translation and parsing
        if "translation" not in config["inference"]:
            logging.warning(
                "'inference.translation' not found in config. Falling back to default translation inference values: "
                "beam_size=4, length_penalty=0.6, max_length_offset=50"
            )
            config["inference"]["translation"] = {
                "beam_size": 4,
                "length_penalty": 0.6,
                "max_length_offset": 50
            }
        if "parsing" not in config["inference"]:
            logging.warning(
                "'inference.parsing' not found in config. Falling back to default parsing inference values: "
                "beam_size=21, length_penalty=0.3, max_length_offset=300"
            )
            config["inference"]["parsing"] = {
                "beam_size": 21,
                "length_penalty": 0.3,
                "max_length_offset": 300
            }

        # Validate checkpoint section keys and defaults
        checkpoint_defaults = {
            "checkpoint_interval_minutes": 10,
            "average_checkpoints": {"base": 5, "big": 20}
        }
        if "checkpoint" not in config:
            logging.warning(
                "'checkpoint' section not found in config. Falling back to default checkpoint values."
            )
            config["checkpoint"] = checkpoint_defaults
        else:
            for key, default in checkpoint_defaults.items():
                if key not in config["checkpoint"]:
                    logging.warning(
                        f"'checkpoint.{key}' not found in config. Falling back to default value: {default}"
                    )
                    config["checkpoint"][key] = default

        logging.info("Configuration loaded and validated successfully.")
        return config

    def log_effective_config(self) -> None:
        """
        Logs the effective configuration for transparency.
        """
        logging.info("Effective configuration:")
        logging.info(self.config)

    def set_random_seeds(self, seed_value: int = 42) -> None:
        """
        Sets random seeds for reproducibility across Python, NumPy, and PyTorch.
        
        Args:
            seed_value: The seed value to use.
        """
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_value)
        logging.info(f"Random seeds set to {seed_value}.")

    def run_experiment(self) -> None:
        """
        Coordinates the entire experiment workflow:
        - Loads data using the DatasetLoader.
        - Instantiates the TransformerModel.
        - Sets up the Trainer and runs training.
        - Runs evaluation using the Evaluation module.
        """
        logging.info("Starting experiment workflow...")

        # Import DatasetLoader from dataset_loader.py
        try:
            from dataset_loader import DatasetLoader
        except ImportError as e:
            logging.critical(f"Failed to import DatasetLoader: {e}")
            sys.exit(1)

        # Determine translation dataset identifier and vocabulary size from config
        translation_data = self.config["data"]["translation"]
        dataset_identifier = translation_data.get("datasets", {}).get("english_german", "WMT14_en_de")
        vocab_size = int(translation_data.get("vocabulary", {}).get("english_german", 37000))

        # Instantiate DatasetLoader (for translation task)
        dataset_loader = DatasetLoader(self.config, task="translation", dataset_name="english_german")
        train_loader, val_loader, test_loader = dataset_loader.load_data()

        # Import TransformerModel from model.py
        try:
            from model import TransformerModel
        except ImportError as e:
            logging.critical(f"Failed to import TransformerModel: {e}")
            sys.exit(1)

        # Instantiate the Transformer model using the "base" configuration by default.
        model = TransformerModel(self.config, model_type="base", vocab_size=vocab_size)
        logging.info(f"Transformer model instantiated with d_model={model.encoder.d_model}")

        # Import Trainer from trainer.py
        try:
            from trainer import Trainer
        except ImportError as e:
            logging.critical(f"Failed to import Trainer: {e}")
            sys.exit(1)

        # Set up the Trainer and run training loop
        trainer = Trainer(model, train_loader, val_loader, self.config)
        logging.info("Beginning training process...")
        trainer.train()
        logging.info("Training process completed.")

        # Import Evaluation from evaluation.py
        try:
            from evaluation import Evaluation
        except ImportError as e:
            logging.critical(f"Failed to import Evaluation: {e}")
            sys.exit(1)

        # Instantiate Evaluation for the translation task using test_loader
        evaluator = Evaluation(model, test_loader, self.config, evaluation_task="translation")
        evaluation_metrics = evaluator.evaluate()
        logging.info(f"Final Evaluation Metrics: {evaluation_metrics}")

        logging.info("Experiment workflow completed successfully.")


if __name__ == "__main__":
    main_runner = Main()
    main_runner.run_experiment()
