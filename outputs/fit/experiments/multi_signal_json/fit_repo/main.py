#!/usr/bin/env python3
"""
main.py

Entry point for training and evaluating the FiT (Flexible Vision Transformer for Diffusion) model.

This script:
  1. Loads the configuration from config.yaml.
  2. Instantiates the DatasetLoader to obtain a DataLoader yielding raw image tensors.
  3. Creates the FiT Model, loads the pretrained VAE weights, and freezes them.
  4. Instantiates the Trainer to commence the training process with the diffusion noise schedule.
  5. After training, instantiates the Evaluation module to run inference and compute evaluation metrics.
  
Author: [Your Name]
Date: [Today's Date]
"""

import os
import yaml
import torch
from typing import Dict

# Import components from the project files.
from dataset_loader import DatasetLoader
from model import Model
from trainer import Trainer
from evaluation import Evaluation


def load_config(config_path: str = "config.yaml") -> Dict:
    """
    Loads the configuration from a YAML file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        Dict: Configuration as a dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file '{config_path}' not found.")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def freeze_vae_parameters(model: Model) -> None:
    """
    Freezes the VAE parameters in the given FiT model to ensure that the pretrained VAE weights remain unchanged.

    Args:
        model (Model): The FiT model instance containing the pretrained VAE.
    """
    if hasattr(model, "vae") and model.vae is not None:
        for param in model.vae.parameters():
            param.requires_grad = False
        print("Pretrained VAE parameters have been frozen.")


def main() -> None:
    """
    Main function to run the experiment:
      - Load configuration.
      - Initialize the DatasetLoader and obtain a DataLoader.
      - Create the FiT Model and freeze its VAE parameters.
      - Initialize the Trainer and run training.
      - After training, initialize the Evaluation module and run evaluation.
    """
    # 1. Load configuration.
    config_path: str = "config.yaml"
    config: Dict = load_config(config_path)
    print("Configuration loaded successfully:")
    print(config)

    # 2. Initialize DatasetLoader to obtain the DataLoader.
    dataset_loader: DatasetLoader = DatasetLoader(config)
    data_loader = dataset_loader.load_data()
    print("DataLoader created successfully.")

    # 3. Instantiate the FiT Model.
    fit_model: Model = Model(config)
    # Load pretrained VAE is called in Model.__init__, now freeze its parameters.
    freeze_vae_parameters(fit_model)
    # Move model to appropriate device.
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fit_model.to(device)
    print(f"FiT Model instantiated and moved to device: {device}")

    # 4. Instantiate Trainer with the model, DataLoader, and configuration; then run training.
    trainer: Trainer = Trainer(fit_model, data_loader, config)
    print("Starting training...")
    trainer.train()
    print("Training completed.")

    # 5. Instantiate Evaluation with the trained model and run evaluation.
    evaluator: Evaluation = Evaluation(fit_model, data_loader, config)
    print("Starting evaluation...")
    eval_results: Dict = evaluator.evaluate()

    # 6. Print evaluation metrics.
    print("\nEvaluation Results:")
    for config_key, metrics in eval_results.items():
        print(f"Configuration: {config_key}")
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name}: {metric_value}")
        print("-" * 30)


if __name__ == "__main__":
    main()
