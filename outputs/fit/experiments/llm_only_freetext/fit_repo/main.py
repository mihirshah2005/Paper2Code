"""
main.py

This is the main entry point for executing the FiT reproduction experiment.
It loads configuration from config.yaml, prepares the dataset, initializes the FiT model,
sets up the training pipeline, and finally runs evaluation.
The code follows the design specifications and uses strong type annotations and explicit default values.
"""

import os
import sys
import yaml
import torch
from typing import Any, Dict

# Import custom modules defined in the project.
from dataset_loader import DatasetLoader
from model import Model
from trainer import Trainer
from evaluation import Evaluation


class Main:
    """
    Main class that orchestrates the FiT experiment pipeline.
    
    It initializes the dataset, model, trainer, and evaluation components
    based on the configuration provided, and runs the training and evaluation procedures.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize Main with configuration, data loader, model, trainer, and evaluation.
        
        Args:
            config (Dict[str, Any]): Experiment configuration loaded from config.yaml.
        """
        self.config: Dict[str, Any] = config

        # Initialize DatasetLoader and obtain the DataLoader.
        self.dataset_loader: DatasetLoader = DatasetLoader(self.config)
        self.data_loader: torch.utils.data.DataLoader = self.dataset_loader.load_data()

        # Initialize the FiT model.
        self.model: Model = Model(self.config)

        # Determine the model variant for selecting the training step parameter.
        # Default variant is "B/2" if not specified.
        self.model_variant: str = str(self.config.get("model", {}).get("variant", "B/2"))
        print(f"Model variant selected: {self.model_variant}")

        # Initialize the Trainer with the model, DataLoader, and the full configuration.
        self.trainer: Trainer = Trainer(self.model, self.data_loader, self.config)

        # Initialize the Evaluation module with the same model and DataLoader.
        self.evaluator: Evaluation = Evaluation(self.model, self.data_loader, self.config)

    def run_experiment(self) -> None:
        """
        Runs the experiment pipeline: training followed by evaluation.
        """
        try:
            print("Starting training...")
            self.trainer.train()
            print("Training completed.")
        except Exception as error:
            print(f"Error during training: {error}")
            sys.exit(1)

        try:
            print("Starting evaluation...")
            eval_results: Dict[str, Any] = self.evaluator.evaluate()
            print("Evaluation completed.")
            print("Final Evaluation Results:")
            for resolution, methods in eval_results.items():
                print(f"Resolution: {resolution}")
                for method, metrics in methods.items():
                    print(f"  Method: {method}")
                    for metric_name, value in metrics.items():
                        print(f"    {metric_name}: {value}")
        except Exception as error:
            print(f"Error during evaluation: {error}")
            sys.exit(1)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path (str): Path to the configuration file. Default is "config.yaml".
    
    Returns:
        A dictionary containing configuration parameters.
    """
    if not os.path.exists(config_path):
        print(f"Configuration file '{config_path}' does not exist.")
        sys.exit(1)
    try:
        with open(config_path, "r") as config_file:
            config: Dict[str, Any] = yaml.safe_load(config_file)
            if config is None:
                raise ValueError("Loaded configuration is empty.")
            return config
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)


def main() -> None:
    """
    Main function to run the FiT reproduction experiment.
    Loads the configuration, initializes the experiment, and runs training and evaluation.
    """
    print("Loading configuration from config.yaml...")
    config: Dict[str, Any] = load_config("config.yaml")
    print("Configuration loaded successfully.")

    # Instantiate the Main class and run the experiment.
    experiment_main = Main(config)
    experiment_main.run_experiment()


if __name__ == "__main__":
    main()
