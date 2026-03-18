"""main.py

This is the main entry point for the Graph Distillation with Eigenbasis Matching (GDEM)
experiment. It loads the configuration from 'config.yaml', initializes the dataset,
the synthetic graph distillation model, the training loop, and finally the evaluation
module. The experiment is designed to distill a small synthetic graph from a large real
graph so that GNNs trained on the synthetic graph yield comparable performance to those
trained on the real graph.

Usage:
    python main.py
"""

import os
import math
import logging
import yaml
import numpy as np
import torch

from dataset_loader import DatasetLoader
from model import Model
from trainer import Trainer
from evaluation import Evaluation

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_config(config_path: str) -> dict:
    """Loads the configuration from a YAML file.

    Args:
        config_path (str): Path to the configuration YAML file.

    Returns:
        dict: Configuration as a dictionary.
    """
    if not os.path.exists(config_path):
        logging.warning(f"Configuration file {config_path} not found. Using default configuration.")
        # Default configuration (should match your config.yaml settings)
        default_config = {
            "training": {
                "epochs": 1500,
                "learning_rate_feat": 1e-05,
                "learning_rate_eigenvecs": 0.01,
                "tau1": 1,  # Setting tau1 to at least 1 for proper alternating updates.
                "tau2": 5,
                "optimizer": "adam",
                "loss_weights": {"alpha": 1.0, "beta": 1.0, "gamma": 1.0},
                "num_runs": 10
            },
            "synth_graph": {
                "compression_ratio": 0.15,  # 0.15% compression ratio
                "r_k": 0.9
            },
            "evaluation": {
                "epochs": 200,
                "learning_rate": 0.01,
                "gnn": {
                    "spatial": {"n_layers": 2, "hidden_units": 256},
                    "spectral": {"polynomial_order": 10, "hidden_units": 256}
                },
                "device": "cpu"
            },
            "dataset": {
                "name": "Pubmed",
                "split_method": "public"
            }
        }
        return default_config
    else:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config


def validate_config(config: dict) -> dict:
    """Validates and potentially adjusts the configuration settings.

    In particular, ensures that 'tau1' (the number of iterations for eigenbasis update)
    is a positive integer. If tau1 is configured as 0 or negative, a warning is logged and
    tau1 is set to 1 to ensure that the synthetic eigenbasis is updated periodically.

    Args:
        config (dict): The configuration dictionary.

    Returns:
        dict: The validated (and possibly adjusted) configuration.
    """
    training_config = config.get("training", {})
    tau1 = int(training_config.get("tau1", 0))
    if tau1 <= 0:
        logging.warning("Configured tau1 is non-positive. Setting tau1 to 1 to ensure eigenbasis updates occur.")
        training_config["tau1"] = 1
    config["training"] = training_config
    return config


def main() -> None:
    """Main entry point of the experiment pipeline.

    It loads configuration, initializes dataset, model, trainer, and evaluation modules,
    then runs the training loop and evaluates the models.
    """
    # Load configuration from config.yaml
    config_path: str = "config.yaml"
    config: dict = load_config(config_path)
    config = validate_config(config)

    # Set device based on configuration (default to "cpu")
    eval_config: dict = config.get("evaluation", {})
    device_str: str = eval_config.get("device", "cpu")
    device: torch.device = torch.device(device_str)

    # Set random seeds for reproducibility.
    seed: int = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    logging.info("Starting Graph Distillation Experiment (GDEM).")
    logging.info(f"Using device: {device_str}")

    # Load the dataset and compute necessary graph information.
    logging.info("Loading dataset...")
    dataset_loader = DatasetLoader(config)
    graph_data = dataset_loader.load_data()

    # Compute synthetic graph size based on the real number of nodes and compression ratio.
    num_nodes: int = graph_data.adjacency.shape[0]
    compression_ratio: float = float(config.get("synth_graph", {}).get("compression_ratio", 0.15))
    # compression_ratio is expressed as a percentage.
    synthetic_size: int = math.ceil(num_nodes * (compression_ratio / 100.0))
    synthetic_size = max(synthetic_size, 1)
    logging.info(f"Real graph has {num_nodes} nodes; synthetic graph will have {synthetic_size} nodes "
                 f"(compression_ratio: {compression_ratio}%).")

    # Determine feature dimension and set K (number of eigenvectors = synthetic_size).
    feature_dim: int = graph_data.node_features.shape[1]
    K: int = synthetic_size  # As per design: K = K1 + K2 and K is set to synthetic_size.
    logging.info(f"Feature dimension: {feature_dim}; Using K = {K} eigenvectors for matching.")

    # Build model parameters.
    model_params: dict = {
        "synthetic_size": synthetic_size,
        "feature_dim": feature_dim,
        "K": K,
        "real_eigenvalues": graph_data.eigenvalues,
        "loss_weights": config.get("training", {}).get("loss_weights", {"alpha": 1.0, "beta": 1.0, "gamma": 1.0}),
        "device": device_str
    }

    # Initialize the synthetic graph distillation model.
    logging.info("Initializing the synthetic graph distillation model...")
    distillation_model = Model(model_params)
    distillation_model.to(device)

    # Initialize Trainer which handles the training loop with alternating updates.
    logging.info("Initializing trainer...")
    trainer_obj = Trainer(distillation_model, graph_data, config)
    logging.info("Starting training...")
    trainer_obj.train()
    logging.info("Training phase completed.")

    # Run evaluation on the distilled synthetic graph.
    logging.info("Starting evaluation of synthetic graph across candidate GNN architectures...")
    evaluator = Evaluation(distillation_model, graph_data, config)
    eval_results: dict = evaluator.evaluate()
    logging.info("Evaluation completed.")

    # Output evaluation results.
    print("\nFinal Evaluation Results:")
    for arch, res in eval_results.items():
        print(f"Architecture: {arch}")
        print(f"    Mean Accuracy: {res['mean_accuracy']:.4f}")
        print(f"    Std Accuracy: {res['std_accuracy']:.4f}")
        print(f"    Accuracies per run: {res['accuracies']}")

    logging.info("Experiment finished successfully.")


if __name__ == "__main__":
    main()
