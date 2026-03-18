#!/usr/bin/env python3
"""
main.py

This file orchestrates the end-to-end training and evaluation of the Transformer model for
either translation or parsing tasks, following the methodology described in
"Attention Is All You Need." It loads the configuration from config.yaml,
initializes the DatasetLoader, TransformerModel, Trainer, and Evaluation modules, applies
explicit weight initialization, and runs the training and evaluation processes.

Usage:
    python main.py --config config.yaml --task translation
    python main.py --config config.yaml --task parsing --parsing_mode wsj_only

Author: [Your Name]
Date: [Date]
"""

import argparse
import os
import random
import logging
import yaml
import math
import numpy as np
from typing import Any

import torch
import torch.nn as nn

# Import project modules
from dataset_loader import DatasetLoader
from model import TransformerModel
from trainer import Trainer
from evaluation import Evaluation

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def init_weights(module: nn.Module) -> None:
    """
    Initialize weights for linear and embedding layers using Xavier (Glorot) uniform initialization.
    Biases (if any) are initialized to zero.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.xavier_uniform_(module.weight)


class Main:
    """
    Main class that coordinates configuration loading, dataset preparation,
    model instantiation, training, and evaluation.
    
    Attributes:
        config: Dictionary loaded from the YAML configuration file.
        task: Task type ("translation" or "parsing").
        dataset_key: For translation tasks, specifies which dataset key to use.
        parsing_mode: For parsing tasks, selects the vocabulary mode (e.g., "wsj_only").
        train_loader, val_loader, test_loader: PyTorch DataLoader objects.
        model: An instance of TransformerModel.
        device: Torch device selected ("cuda" if available, else "cpu").
    """
    def __init__(self, config: dict, task: str, dataset_key: str = None, parsing_mode: str = None) -> None:
        self.config = config
        self.task = task.lower()
        # Set default seed for reproducibility
        self.seed: int = 42
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Instantiate DatasetLoader based on task type.
        if self.task == "translation":
            # Use provided dataset_key or default to "english_german"
            self.dataset_key: str = dataset_key if dataset_key is not None else "english_german"
            self.dataset_loader = DatasetLoader(self.config, task="translation", dataset_key=self.dataset_key)
        elif self.task == "parsing":
            self.parsing_mode: str = parsing_mode if parsing_mode is not None else "wsj_only"
            self.dataset_loader = DatasetLoader(self.config, task="parsing", parsing_mode=self.parsing_mode)
        else:
            raise ValueError(f"Unsupported task type: {self.task}. Supported types are 'translation' and 'parsing'.")

        # Load raw datasets and create DataLoaders (train, validation, test)
        self.train_loader, self.val_loader, self.test_loader = self.dataset_loader.load_data()

        # Determine vocabulary size from DatasetLoader.
        if self.task == "translation":
            # For translation, use the shared vocabulary size from the config.
            self.vocab_size: int = self.dataset_loader.translation_vocab_size
        else:
            self.vocab_size: int = self.dataset_loader.parsing_vocab_size

        # Instantiate the TransformerModel.
        # For both tasks, the source and target vocabulary sizes are the same.
        self.model = TransformerModel(self.config, src_vocab_size=self.vocab_size, tgt_vocab_size=self.vocab_size)
        # Apply explicit weight initialization across all learnable parameters.
        self.model.apply(init_weights)
        # Select device and move model to that device.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        LOGGER.info(f"Initialized Transformer model for task '{self.task}' with vocab size {self.vocab_size} on device {self.device}.")

    def run_experiment(self) -> None:
        """
        Carries out the training and evaluation phases.
        """
        LOGGER.info("Starting training process...")
        trainer = Trainer(self.model, self.train_loader, self.val_loader, self.config)
        trainer.train()  # This runs the training loop with gradient clipping, checkpointing, etc.

        LOGGER.info("Training complete. Starting evaluation phase...")
        evaluator = Evaluation(self.model, self.test_loader, self.config)
        metrics: dict = evaluator.evaluate()

        LOGGER.info("Final Evaluation Metrics:")
        for metric_name, metric_value in metrics.items():
            LOGGER.info(f"  {metric_name}: {metric_value:.2f}")

        LOGGER.info("Experiment finished successfully.")


def load_config(config_path: str) -> dict:
    """
    Loads and returns the configuration from a given YAML file.
    
    Args:
        config_path: Path to the YAML configuration file.
    
    Returns:
        A dictionary containing configuration parameters.
    
    Raises:
        FileNotFoundError: If the configuration file is not found.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file '{config_path}' not found.")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.
    
    Returns:
        An argparse.Namespace object with parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train and evaluate the Transformer model.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration YAML file.")
    parser.add_argument("--task", type=str, choices=["translation", "parsing"], default="translation", help="Task type: translation or parsing.")
    parser.add_argument("--dataset_key", type=str, default=None, help="Dataset key for translation tasks (e.g., 'english_german' or 'english_french').")
    parser.add_argument("--parsing_mode", type=str, default=None, help="Parsing mode (e.g., 'wsj_only' or 'semi_supervised').")
    return parser.parse_args()


def main() -> None:
    """
    Main function for running the experiment. It loads the configuration, initializes the Main class,
    and triggers the experiment workflow.
    """
    args = parse_arguments()
    config: dict = load_config(args.config)

    # Set the evaluation task in configuration if not already set.
    if "evaluation_task" not in config:
        config["evaluation_task"] = args.task

    LOGGER.info(f"Configuration loaded from {args.config}.")
    LOGGER.info(f"Task selected: {args.task}.")

    # Instantiate and run the experiment.
    experiment = Main(config, task=args.task, dataset_key=args.dataset_key, parsing_mode=args.parsing_mode)
    experiment.run_experiment()


if __name__ == "__main__":
    main()
