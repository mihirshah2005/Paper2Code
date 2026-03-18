"""
main.py

Main entry point for the Graph Distillation with Eigenbasis Matching (GDEM) experiment.
This script coordinates the loading of configuration, dataset preprocessing, model initialization,
training (synthetic graph distillation), synthetic graph reconstruction, and evaluation using various GNN architectures.

Steps:
  1. Load configuration from config.yaml (or use default values).
  2. Instantiate DatasetLoader and load the graph data.
  3. Initialize the Model with synthetic node features and eigenbasis.
  4. Set up the Trainer and run the training loop for the synthetic graph distillation.
  5. Reconstruct the synthetic graph (adjacency matrix and node features) from the learned parameters.
  6. Instantiate the Evaluation module and evaluate the distilled synthetic graph.
  7. Log and print the evaluation results.

Required external packages:
    PyYAML, numpy, torch, scipy, networkx
"""

import os
import yaml
import torch
from dataset_loader import DatasetLoader, GraphData
from model import Model
from trainer import Trainer
from evaluation import Evaluation


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Load configuration settings from a YAML file.
    If the file does not exist, use default configuration values.

    Args:
        config_path (str): Path to the config.yaml file.

    Returns:
        dict: Configuration dictionary.
    """
    default_config = {
        "training": {
            "epochs": 1500,
            "learning_rate_feat": 1e-05,
            "learning_rate_eigenvecs": 0.01,
            "tau1": 0,
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
            "gnn": {
                "spatial": {"n_layers": 2, "hidden_units": 256},
                "spectral": {"polynomial_order": 10, "hidden_units": 256},
                "epochs": 200,
                "learning_rate": 0.01,
                "num_runs": 10
            }
        },
        "dataset": {
            "name": "Pubmed",
            "split_method": "public"
        }
    }
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            # Ensure that if any key is not set in the file, defaults are used.
            # Recursively merge default_config into config.
            def merge_configs(default: dict, custom: dict) -> dict:
                for key, value in default.items():
                    if key not in custom:
                        custom[key] = value
                    elif isinstance(value, dict):
                        custom[key] = merge_configs(value, custom.get(key, {}))
                return custom
            config = merge_configs(default_config, config)
            print("Configuration loaded successfully from config.yaml.")
        except Exception as e:
            print(f"Error loading configuration file: {e}")
            print("Using default configuration.")
            config = default_config
    else:
        print("config.yaml not found. Using default configuration.")
        config = default_config
    return config


def main() -> None:
    """
    Main function to run the entire GDEM pipeline:
      - Load configuration
      - Load dataset and preprocess graph
      - Initialize the synthetic graph distillation Model
      - Train the model using the Trainer
      - Reconstruct the synthetic graph
      - Evaluate the distilled synthetic graph with different GNN architectures
    """
    # Load configuration from file or default
    config = load_config("config.yaml")

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # Instantiate DatasetLoader and load the graph data
    dataset_loader = DatasetLoader(config)
    graph_data: GraphData = dataset_loader.load_data()
    print("Dataset loaded successfully.")
    print(f"Graph details: Number of nodes = {graph_data.adjacency.shape[0]}, "
          f"Feature dimension = {graph_data.node_features.shape[1]}")

    # Retrieve synthetic graph size and feature dimension from graph_data
    N_syn: int = graph_data.synthetic_size
    feature_dim: int = graph_data.node_features.shape[1]
    print(f"Initializing model with synthetic graph size (N'): {N_syn} and feature dimension: {feature_dim}")

    # Instantiate the Model with configuration, synthetic graph size, and feature dimension
    model_instance = Model(config=config, N_syn=N_syn, d=feature_dim, device=device)
    model_instance.to(device)

    # Instantiate Trainer with Model, GraphData, and configuration
    trainer_instance = Trainer(model=model_instance, graph_data=graph_data, config=config)
    print("Starting synthetic graph distillation training...")
    
    # Run the training procedure; returns reconstructed synthetic adjacency and final synthetic node features.
    synthetic_adjacency, synthetic_features = trainer_instance.train()
    print("Training complete.")
    print(f"Final Synthetic Adjacency Matrix shape: {synthetic_adjacency.shape}")
    print(f"Final Synthetic Node Features shape: {synthetic_features.shape}")

    # Instantiate Evaluation with the trained Model, full GraphData, and evaluation configuration
    evaluation_instance = Evaluation(model=model_instance, data=graph_data, eval_config=config)
    print("Starting evaluation of distilled synthetic graph using various GNN architectures...")
    
    # Run evaluation; returns a dictionary of evaluation metrics (mean accuracy and std per architecture)
    evaluation_results = evaluation_instance.evaluate()
    print("Evaluation completed. Results:")
    for arch, metrics in evaluation_results.items():
        print(f"Architecture {arch}: Mean Accuracy = {metrics['mean']:.4f}, Std = {metrics['std']:.4f}")


if __name__ == "__main__":
    main()
