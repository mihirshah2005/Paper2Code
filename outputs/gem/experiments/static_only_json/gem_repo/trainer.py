"""
trainer.py

This module implements the Trainer class that orchestrates the training loop for synthetic graph distillation 
using the GDEM methodology. The Trainer alternates updates between the synthetic eigenbasis (U_syn) and synthetic 
node features (X_syn) based on a scheduling parameter (tau1, tau2) provided in the configuration. It uses the Adam 
optimizer for both parameter groups. Finally, after training, it reconstructs the synthetic graph (adjacency 
and Laplacian) from the learned synthetic eigenbasis and the real graph spectrum.

Required Packages:
    torch==1.9.0
    numpy==1.21.0
"""

import torch
import torch.optim as optim
from torch import nn, Tensor
from typing import Tuple, Dict, Any

# Import Model and GraphData from the respective modules
# It is assumed that the project structure places model.py and dataset_loader.py in the same package
from model import Model
from dataset_loader import GraphData


def compute_total_variation(features: Tensor) -> float:
    """Compute a simple total variation (TV) metric for synthetic node features.
    
    TV is computed as the mean absolute deviation from the global mean.
    
    Args:
        features (Tensor): Synthetic node features (shape: [N_syn, d]).
        
    Returns:
        float: The computed total variation as a scalar.
    """
    global_mean: Tensor = torch.mean(features, dim=0, keepdim=True)
    tv: Tensor = torch.mean(torch.abs(features - global_mean))
    return tv.item()


class Trainer:
    """
    Trainer class orchestrates the training loop for synthetic graph distillation using the GDEM method.
    
    Attributes:
        model (Model): Instance of the Model wrapper for synthetic graph distillation.
        graph_data (GraphData): Loaded real graph data containing adjacency, features, labels, Laplacian,
                                eigenvalues, eigenvectors, and synthetic graph size.
        config (dict): Configuration dictionary including training parameters and scheduling values.
        epochs (int): Number of training epochs.
        tau1 (int): Number of iterations in the cycle allocated for updating the synthetic eigenbasis.
        tau2 (int): Number of iterations in the cycle allocated for updating the synthetic node features.
        optimizer_eigen (optim.Optimizer): Adam optimizer for the synthetic eigenbasis parameters.
        optimizer_feat (optim.Optimizer): Adam optimizer for the synthetic node feature parameters.
        device (torch.device): Device (CPU or GPU) to run the training on.
    """
    def __init__(self, model: Model, graph_data: GraphData, config: Dict[str, Any]) -> None:
        """
        Initialize the Trainer with the given Model, GraphData and configuration.
        
        Args:
            model (Model): The synthetic graph distillation model.
            graph_data (GraphData): The real graph data loaded via DatasetLoader.
            config (dict): The configuration dictionary, typically loaded from config.yaml.
        """
        self.model: Model = model
        self.graph_data: GraphData = graph_data
        self.config: Dict[str, Any] = config

        # Set device from model.device if available, otherwise default to CPU.
        self.device: torch.device = self.model.device

        # Retrieve training parameters from configuration.
        training_config: Dict[str, Any] = config.get("training", {})
        self.epochs: int = int(training_config.get("epochs", 1500))
        self.tau1: int = int(training_config.get("tau1", 0))
        self.tau2: int = int(training_config.get("tau2", 5))
        self.cycle: int = self.tau1 + self.tau2 if (self.tau1 + self.tau2) > 0 else 1

        # Create separate Adam optimizers for synthetic eigenbasis and synthetic node features.
        self.optimizer_eigen: optim.Optimizer = optim.Adam(
            [self.model.U_syn], lr=self.model.lr_eigenvecs
        )
        self.optimizer_feat: optim.Optimizer = optim.Adam(
            [self.model.X_syn], lr=self.model.lr_feat
        )

        # Precompute the target real eigenbasis to match.
        # We take the first N_syn rows of the real eigenvectors and the first model.K columns.
        N_syn: int = self.graph_data.synthetic_size
        K: int = self.model.K
        real_eigen_np = self.graph_data.eigenvectors[:N_syn, :K]
        self.real_eigen_target: Tensor = torch.tensor(
            real_eigen_np, dtype=torch.float32, device=self.device
        )

        # Precompute the real category-level representation as the global mean of real node features.
        real_features_np = self.graph_data.node_features
        self.real_cat_repr: Tensor = torch.tensor(
            real_features_np, dtype=torch.float32, device=self.device
        ).mean(dim=0)

    def train(self) -> Tuple[Tensor, Tensor]:
        """
        Execute the training loop for synthetic graph distillation.
        
        The loop alternates between updating the synthetic eigenbasis and synthetic node features based 
        on the schedule defined by tau1 and tau2. Loss is computed as a weighted sum of eigenbasis matching 
        loss (L_e), discrimination constraint loss (L_d), and orthogonality regularization loss (L_o).
        
        After training, the synthetic graph is reconstructed using the learned synthetic eigenbasis and the 
        real spectrum, and the function returns the synthetic adjacency matrix and synthetic node features.
        
        Returns:
            Tuple[Tensor, Tensor]: A tuple (A_syn, X_syn) where:
                - A_syn (Tensor): Reconstructed synthetic adjacency matrix.
                - X_syn (Tensor): Final synthetic node features.
        """
        print("Starting training loop for synthetic graph distillation...")
        for epoch in range(self.epochs):
            self.model.train()  # set model to training mode

            # Forward pass: obtain synthetic node features and synthetic eigenbasis.
            X_syn_current, U_syn_current = self.model.forward()

            # Compute synthetic category-level representation as global average of synthetic features.
            synthetic_cat_repr: Tensor = X_syn_current.mean(dim=0)

            # Compute total composite loss via the model's compute_loss method.
            loss: Tensor = self.model.compute_loss(
                real_eigen=self.real_eigen_target,
                real_cat_repr=self.real_cat_repr,
                synthetic_cat_repr=synthetic_cat_repr
            )

            # Determine update branch based on alternating schedule.
            current_cycle_index: int = epoch % self.cycle
            if current_cycle_index < self.tau1:
                # Update synthetic eigenbasis only.
                self.optimizer_eigen.zero_grad()
                self.optimizer_feat.zero_grad()  # ensure features gradients are cleared
                loss.backward()
                self.optimizer_eigen.step()
                # Optionally, clear gradients for node features so they are not updated.
                self.model.X_syn.grad = None
                update_type: str = "Eigenbasis Update"
            else:
                # Update synthetic node features only.
                self.optimizer_feat.zero_grad()
                self.optimizer_eigen.zero_grad()  # ensure eigenbasis gradients are cleared
                loss.backward()
                self.optimizer_feat.step()
                # Optionally, clear gradients for eigenbasis so they are not updated.
                self.model.U_syn.grad = None
                update_type: str = "Node Feature Update"

            # Optionally, compute total variation (TV) metric for synthetic node features.
            tv_metric: float = compute_total_variation(self.model.X_syn)

            # Log training progress every 50 epochs.
            if (epoch + 1) % 50 == 0 or epoch == 0:
                print(
                    f"Epoch [{epoch + 1}/{self.epochs}] | Loss: {loss.item():.6f} | {update_type} | TV: {tv_metric:.6f}"
                )

        print("Training complete. Reconstructing synthetic graph...")

        # Reconstruct the synthetic graph using the learned synthetic eigenbasis and real eigenvalues.
        # Use the first K eigenvalues from the real graph as the replicated spectrum.
        real_eigenvalues_np = self.graph_data.eigenvalues[:self.model.K]
        real_eigenvalues_target: Tensor = torch.tensor(
            real_eigenvalues_np, dtype=torch.float32, device=self.device
        )
        A_syn, L_syn = self.model.reconstruct_graph(real_eigenvalues=real_eigenvalues_target)

        print("Synthetic graph reconstruction complete.")
        return A_syn, self.model.X_syn


# Example usage of the Trainer class if this file is executed directly.
if __name__ == "__main__":
    import yaml
    import os
    from dataset_loader import DatasetLoader

    # Load configuration from config.yaml if available; otherwise, use default settings.
    config_file: str = "config.yaml"
    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            config_data: dict = yaml.safe_load(f)
    else:
        # Default configuration if config.yaml is not found.
        config_data = {
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
                "compression_ratio": 0.15,
                "r_k": 0.9
            },
            "evaluation": {
                "gnn": {
                    "spatial": {"n_layers": 2, "hidden_units": 256},
                    "spectral": {"polynomial_order": 10, "hidden_units": 256}
                }
            },
            "dataset": {
                "name": "Pubmed",
                "split_method": "public"
            }
        }

    # Instantiate DatasetLoader to load the graph data.
    dataset_loader = DatasetLoader(config_data)
    graph_data: GraphData = dataset_loader.load_data()

    # Create the Model instance using synthetic graph size and feature dimension.
    N_syn: int = graph_data.synthetic_size
    d: int = graph_data.node_features.shape[1]
    model_instance = Model(config=config_data, N_syn=N_syn, d=d)

    # Instantiate the Trainer with the model, graph data, and configuration.
    trainer_instance = Trainer(model=model_instance, graph_data=graph_data, config=config_data)
    
    # Run the training loop.
    synthetic_adjacency, synthetic_features = trainer_instance.train()

    # Log final synthetic graph dimensions.
    print(f"Final Synthetic Adjacency Matrix shape: {synthetic_adjacency.shape}")
    print(f"Final Synthetic Node Features shape: {synthetic_features.shape}")
