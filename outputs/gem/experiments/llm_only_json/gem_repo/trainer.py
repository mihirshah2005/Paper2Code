"""
trainer.py

This module implements the Trainer class which orchestrates the training loop for 
Graph Distillation with Eigenbasis Matching (GDEM). It enforces a strict alternating 
update mechanism between the synthetic eigenbasis parameters and the synthetic node 
feature parameters using two separate Adam optimizers. At each iteration, only one 
parameter group is updated as dictated by the alternating schedule (tau1 and tau2).

The Trainer computes two separate loss components:
  - Eigenbasis Matching Loss (loss_eigen): combines the eigenbasis matching loss (Lₑ)
    and the orthogonality regularization loss (Lₒ), used to update the synthetic eigenbasis.
  - Discrimination Constraint Loss (loss_feature): based on a cosine-similarity loss (L_d) 
    between the category-level representations of the real and synthetic graphs, and is 
    used to update the synthetic node features.

Only the active branch’s loss is allowed to backpropagate (the inactive one is detached)
to prevent gradient leakage.
"""

import logging
import numpy as np
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.optim import Adam

# Import the Model class from our model module.
from model import Model

# Set up module-level logger.
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.hasHandlers():
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def compute_category_repr(features: torch.Tensor, labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """
    Computes the category-level representations by averaging feature vectors for each class.
    
    Args:
        features (torch.Tensor): Feature matrix of shape [num_nodes, feature_dim].
        labels (np.ndarray): Array of node labels of shape [num_nodes].
        num_classes (int): Total number of unique classes.
    
    Returns:
        torch.Tensor: Tensor of shape [num_classes, feature_dim] containing, for each class,
                      the average feature vector.
    """
    feature_dim = features.size(1)
    # Initialize accumulators.
    class_sum = torch.zeros(num_classes, feature_dim, device=features.device)
    class_count = torch.zeros(num_classes, device=features.device)
    
    labels_tensor = torch.tensor(labels, dtype=torch.long, device=features.device)
    for c in range(num_classes):
        mask = (labels_tensor == c)
        if mask.sum() > 0:
            class_sum[c] = features[mask].mean(dim=0)
            class_count[c] = mask.sum()
        else:
            # In case a class is missing, assign zero vector.
            class_sum[c] = torch.zeros(feature_dim, device=features.device)
            class_count[c] = 1.0  # To avoid division by zero.
    return class_sum


def generate_synthetic_labels(real_labels: np.ndarray, num_synth_nodes: int) -> np.ndarray:
    """
    Generates synthetic labels for synthetic nodes by sampling from the distribution of real labels.
    
    Args:
        real_labels (np.ndarray): Array of real graph labels.
        num_synth_nodes (int): Number of synthetic nodes.
    
    Returns:
        np.ndarray: Array of synthetic labels of length num_synth_nodes.
    """
    np.random.seed(42)
    unique, counts = np.unique(real_labels, return_counts=True)
    probabilities = counts / counts.sum()
    synth_labels = np.random.choice(unique, size=num_synth_nodes, p=probabilities)
    return synth_labels


class Trainer:
    """
    Trainer class orchestrates the training loop of the GDEM model.
    
    Public Methods:
        __init__(model: Model, data: GraphData, config: dict): Initializes the trainer.
        train() -> Tuple[torch.Tensor, torch.Tensor]: Runs the training loop and returns the 
            reconstructed synthetic adjacency and Laplacian matrices.
    """
    
    def __init__(self, model: Model, data, config: dict) -> None:
        """
        Initializes the Trainer.
        
        Args:
            model (Model): An instance of the GDEM Model.
            data (GraphData): GraphData object containing adjacency, node features, labels,
                              eigenvalues, eigenvectors, and split indices.
            config (dict): Configuration dictionary (e.g., loaded from config.yaml).
        """
        self.model = model
        self.data = data
        self.config = config
        
        # Determine device.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Extract training configuration.
        training_config = self.config.get("training", {})
        self.epochs: int = int(training_config.get("epochs", 1500))
        self.lr_feat: float = float(training_config.get("learning_rate_feat", 1e-05))
        self.lr_eigenvecs: float = float(training_config.get("learning_rate_eigenvecs", 0.01))
        self.tau1: int = int(training_config.get("tau1", 0))
        self.tau2: int = int(training_config.get("tau2", 5))
        
        # Enforce tau1 > 0 to maintain the strict alternating update mechanism.
        if self.tau1 <= 0:
            logger.warning("tau1 is non-positive. Overriding tau1 to 1 to enforce proper alternating updates.")
            self.tau1 = 1
        
        self.cycle: int = self.tau1 + self.tau2
        
        # Create two optimizers: one for eigenbasis (U_synth) and one for node features (X_synth).
        self.optimizer_eigen = Adam([self.model.U_synth], lr=self.lr_eigenvecs)
        self.optimizer_feat = Adam([self.model.X_synth], lr=self.lr_feat)
        
        # Prepare the target real eigenbasis.
        # We assume the synthetic node count equals model.num_synth_nodes and also defines K.
        num_real_nodes = self.data.adjacency.shape[0]
        num_synth_nodes = self.model.num_synth_nodes
        total_k: int = num_synth_nodes  # K is set to number of synthetic nodes.
        if self.data.eigenvectors.shape[0] < num_synth_nodes:
            raise ValueError("Insufficient rows in real eigenvectors compared to synthetic node count.")
        real_eigen_np = self.data.eigenvectors[:num_synth_nodes, :total_k]
        self.real_eigen = torch.tensor(real_eigen_np, dtype=torch.float32, device=self.device)
        
        # Compute real category-level representations H from the real graph's node features and labels.
        real_features_tensor = torch.tensor(self.data.node_features, dtype=torch.float32, device=self.device)
        real_labels_np = self.data.labels
        num_classes = int(np.max(real_labels_np)) + 1
        self.num_classes = num_classes
        self.H_real = compute_category_repr(real_features_tensor, real_labels_np, num_classes)
        
        # Generate synthetic labels for the synthetic graph.
        synth_labels_np = generate_synthetic_labels(real_labels_np, num_synth_nodes)
        self.synthetic_labels = synth_labels_np  # kept as numpy array.
        
        logger.info("Trainer initialized with epochs: %d, lr_feat: %f, lr_eigenvecs: %f, tau1: %d, tau2: %d, cycle: %d",
                    self.epochs, self.lr_feat, self.lr_eigenvecs, self.tau1, self.tau2, self.cycle)

    def compute_loss_dict(self, update_branch: str) -> Dict[str, torch.Tensor]:
        """
        Computes the separate loss components for the eigenbasis branch and the feature branch.
        
        The losses are computed as follows:
          - Eigenbasis Matching Loss (Lₑ):
              L_e = || P_real - P_synth ||_F^2, where P_real = (real_eigen @ real_eigen^T)
                      and P_synth = (U_synth_normalized @ U_synth_normalized^T)
          - Orthogonality Loss (Lₒ):
              L_o = || (U_synth_normalized^T U_synth_normalized - I) ||_F^2
          - Discrimination Constraint Loss (L_d):
              L_d = 1 - mean(cosine_similarity(H_real, H_prime))
              where H_prime is computed from the synthetic node features and synthetic labels.
        
        The total losses for each branch are:
          loss_eigen = alpha * L_e + gamma * L_o
          loss_feature = beta * L_d
        
        Args:
            update_branch (str): Indicates the active branch ("eigen" or "feature").
            
        Returns:
            dict: A dictionary with keys "loss_eigen" and "loss_feature". The inactive branch loss is 
                  detached to prevent gradient propagation.
        """
        model_out = self.model.forward()
        U_synth_norm = model_out["U_synth"]  # shape: [N_synth, K]
        
        # Eigenbasis Matching Loss (Lₑ)
        P_real = self.real_eigen @ self.real_eigen.t()  # [N_synth, N_synth]
        P_synth = U_synth_norm @ U_synth_norm.t()         # [N_synth, N_synth]
        L_e = torch.norm(P_real - P_synth, p="fro") ** 2
        
        # Orthogonality Loss (Lₒ)
        G = U_synth_norm.t() @ U_synth_norm               # [K, K]
        identity = torch.eye(self.model.K, device=G.device)
        L_o = torch.norm(G - identity, p="fro") ** 2
        
        loss_eigen = self.model.alpha * L_e + self.model.gamma * L_o
        
        # Discrimination Constraint Loss (L_d)
        X_synth = self.model.forward()["X_synth"]          # [N_synth, feature_dim]
        H_prime = compute_category_repr(X_synth, self.synthetic_labels, self.num_classes)
        H_real_normalized = F.normalize(self.H_real, p=2, dim=1)
        H_prime_normalized = F.normalize(H_prime, p=2, dim=1)
        cosine_sim = F.cosine_similarity(H_real_normalized, H_prime_normalized, dim=1)
        L_d = 1.0 - torch.mean(cosine_sim)
        
        loss_feature = self.model.beta * L_d
        
        # Detach the inactive branch's loss.
        if update_branch == "eigen":
            loss_feature = loss_feature.detach()
        elif update_branch == "feature":
            loss_eigen = loss_eigen.detach()
            L_o = L_o.detach()
        
        return {"loss_eigen": loss_eigen, "loss_feature": loss_feature}

    def train(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs the training loop for the configured number of epochs using the strict alternating update
        mechanism. At each iteration, the active branch is determined by the cycle defined by tau1 and tau2.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The final reconstructed synthetic adjacency matrix and 
            synthetic Laplacian matrix.
        """
        self.model.train()
        
        for epoch in range(self.epochs):
            # Determine the update branch based on the current iteration within the cycle.
            cycle_step = epoch % self.cycle
            if cycle_step < self.tau1:
                update_branch = "eigen"
            else:
                update_branch = "feature"
            
            # Zero gradients for both optimizer groups.
            self.optimizer_eigen.zero_grad()
            self.optimizer_feat.zero_grad()
            
            # Compute the separated loss components.
            loss_dict = self.compute_loss_dict(update_branch)
            if update_branch == "eigen":
                loss_active = loss_dict["loss_eigen"]
            else:
                loss_active = loss_dict["loss_feature"]
            
            # Backpropagate only the active loss.
            loss_active.backward()
            
            # Step the corresponding optimizer.
            if update_branch == "eigen":
                self.optimizer_eigen.step()
            else:
                self.optimizer_feat.step()
            
            if epoch % 100 == 0 or epoch == self.epochs - 1:
                total_loss = loss_dict["loss_eigen"] + loss_dict["loss_feature"]
                logger.info("Epoch %d: update_branch=%s, loss_eigen=%.4f, loss_feature=%.4f, total_loss=%.4f",
                            epoch, update_branch,
                            loss_dict["loss_eigen"].item(),
                            loss_dict["loss_feature"].item(),
                            total_loss.item())
        
        # After training, reconstruct the synthetic graph.
        # Prepare target real eigenvalues: select the first K elements.
        total_k = self.model.K
        real_eigenvalues_np = self.data.eigenvalues[:total_k]
        real_eigenvalues = torch.tensor(real_eigenvalues_np, dtype=torch.float32, device=self.device)
        
        self.model.eval()
        with torch.no_grad():
            A_prime, L_prime = self.model.reconstruct_graph(real_eigenvalues)
        
        logger.info("Training completed. Synthetic graph reconstructed.")
        return A_prime, L_prime


if __name__ == "__main__":
    # Example usage of the Trainer module.
    import yaml
    from dataset_loader import DatasetLoader

    # Load configuration from 'config.yaml'
    config_path = "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Initialize the dataset loader and load the graph data.
    dataset_loader = DatasetLoader(config)
    graph_data = dataset_loader.load_data()

    # Prepare model parameters using configuration and dataset statistics.
    synth_graph_cfg = config.get("synth_graph", {})
    compression_ratio = synth_graph_cfg.get("compression_ratio", 0.15)
    num_synth_nodes = max(1, int(round(compression_ratio * graph_data.adjacency.shape[0])))
    model_params = {
        "num_synth_nodes": num_synth_nodes,
        "feature_dim": graph_data.node_features.shape[1],
        "loss_weights": config.get("training", {}).get("loss_weights", {"alpha": 1.0, "beta": 1.0, "gamma": 1.0}),
        "learning_rate_eigenvecs": config.get("training", {}).get("learning_rate_eigenvecs", 0.01),
        "learning_rate_feat": config.get("training", {}).get("learning_rate_feat", 1e-05),
        "r_k": synth_graph_cfg.get("r_k", 0.9)
    }

    model = Model(model_params)

    # Initialize the Trainer with model, graph data, and configuration.
    trainer = Trainer(model, graph_data, config)

    # Run the training process.
    A_prime, L_prime = trainer.train()

    # For demonstration, print the shapes of the reconstructed matrices.
    print("Reconstructed synthetic Adjacency matrix shape:", A_prime.shape)
    print("Reconstructed synthetic Laplacian matrix shape:", L_prime.shape)
