"""main.py

This is the entry point for the Transformer experiment reproduction. It parses the configuration 
(from config.yaml), validates critical parameters (mode and task), loads the appropriate dataset 
using DatasetLoader, instantiates the TransformerModel according to the configuration (base or big), 
and then either trains the model using Trainer or evaluates it using Evaluation.

Usage examples:
  To train:
    python main.py --mode train --task translation --config config.yaml --model_type base
  To evaluate a pretrained model:
    python main.py --mode eval --task translation --config config.yaml --checkpoint_path path/to/checkpoint.pt --model_type base

Author: [Your Name]
Date: [Current Date]
"""

import os
import sys
import argparse
import yaml
import logging
import torch

# Import project modules
from dataset_loader import DatasetLoader
from model import TransformerModel
from trainer import Trainer
from evaluation import Evaluation

# Setup the main logger for the experiment.
logger = logging.getLogger("Main")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Transformer Experiment: Train or Evaluate the model based on config."
    )
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to the configuration YAML file (default: config.yaml)")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"],
                        help="Mode of operation: train or eval (default: train)")
    parser.add_argument("--task", type=str, default="translation", choices=["translation", "parsing"],
                        help="Experiment task: translation or parsing (default: translation)")
    parser.add_argument("--model_type", type=str, default="base", choices=["base", "big"],
                        help="Model type: base or big (default: base)")
    parser.add_argument("--checkpoint_path", type=str, default="",
                        help="Optional path to a pretrained model checkpoint for evaluation (only used in eval mode)")
    return parser.parse_args()

def load_config(config_path: str) -> dict:
    """
    Load and validate configuration from a YAML file.
    
    Args:
        config_path (str): Path to the YAML configuration file.
        
    Returns:
        dict: Configuration dictionary.
    """
    if not os.path.isfile(config_path):
        logger.error(f"Configuration file {config_path} not found.")
        sys.exit(1)
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error reading config file: {e}")
        sys.exit(1)

    # Validate critical keys and set defaults if missing.
    if "training" not in config:
        logger.warning("Missing 'training' section in config; setting default training parameters.")
        config["training"] = {}
    if "model" not in config:
        logger.warning("Missing 'model' section in config; setting default model parameters.")
        config["model"] = {"base": {}, "big": {}}
    if "data" not in config:
        logger.warning("Missing 'data' section in config; setting default data parameters.")
        config["data"] = {}
    if "inference" not in config:
        logger.warning("Missing 'inference' section in config; setting default inference parameters.")
        config["inference"] = {}
    if "checkpoint" not in config:
        logger.warning("Missing 'checkpoint' section in config; setting default checkpoint parameters.")
        config["checkpoint"] = {}
    return config

def get_vocab_size(config: dict, task: str) -> int:
    """
    Determine the vocabulary size from the configuration based on the task.
    
    Args:
        config (dict): Configuration dictionary.
        task (str): Task name ("translation" or "parsing").
    
    Returns:
        int: Vocabulary size.
    """
    if task == "translation":
        # Default vocabulary size for English-German: 37000; for English-French: 32000.
        vocab_info = config.get("data", {}).get("translation", {}).get("vocabulary", {})
        vocab_size = vocab_info.get("english_german", 37000)
    elif task == "parsing":
        vocab_info = config.get("data", {}).get("parsing", {}).get("vocabulary", {})
        vocab_size = vocab_info.get("wsj_only", 16000)
    else:
        logger.error(f"Unsupported task: {task}")
        sys.exit(1)
    return int(vocab_size)

def main():
    # Parse command-line arguments.
    args = parse_args()
    mode: str = args.mode.lower()
    task: str = args.task.lower()
    model_type: str = args.model_type.lower()
    
    # Load configuration.
    config: dict = load_config(args.config)
    logger.info("Configuration loaded successfully.")
    
    # Log key configuration parameters.
    logger.info(f"Mode: {mode}, Task: {task}, Model Type: {model_type}")

    # Initialize DatasetLoader and load data.
    logger.info("Initializing DatasetLoader...")
    dataset_loader = DatasetLoader(config, task=task)
    train_loader, val_loader, test_loader = dataset_loader.load_data()
    logger.info("Data loaded successfully.")
    
    # Determine vocabulary size from the config.
    vocab_size: int = get_vocab_size(config, task)
    logger.info(f"Using vocabulary size: {vocab_size}")
    
    # Instantiate the Transformer model.
    logger.info("Instantiating Transformer model...")
    model = TransformerModel(config, vocab_size=vocab_size, task=task, model_type=model_type)
    logger.info("Transformer model created.")
    
    # Depending on mode, either train the model or run evaluation.
    if mode == "train":
        logger.info("Starting training mode...")
        trainer = Trainer(model, train_loader, val_loader, config)
        trainer.train()
        logger.info("Training completed. Proceeding to evaluation on test set...")
    elif mode == "eval":
        logger.info("Starting evaluation mode...")
        if args.checkpoint_path:
            if os.path.isfile(args.checkpoint_path):
                logger.info(f"Loading pretrained checkpoint from {args.checkpoint_path}...")
                checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
                model.load_state_dict(checkpoint["model_state_dict"])
                logger.info("Pretrained checkpoint loaded.")
            else:
                logger.warning(f"Checkpoint file {args.checkpoint_path} not found; proceeding with untrained model.")
        else:
            logger.warning("No pretrained checkpoint specified; evaluation will use current model parameters.")
    else:
        logger.error(f"Unsupported mode: {mode}. Supported modes are 'train' and 'eval'.")
        sys.exit(1)
    
    # Run evaluation on the test set.
    logger.info("Initializing Evaluation module...")
    evaluator = Evaluation(model, test_loader, config)
    metrics: dict = evaluator.evaluate()
    logger.info("Evaluation completed.")
    logger.info(f"Final Evaluation Metrics: {metrics}")

if __name__ == "__main__":
    main()
