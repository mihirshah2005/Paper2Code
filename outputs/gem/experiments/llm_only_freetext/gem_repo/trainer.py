"""
trainer.py

This module implements the Trainer class that orchestrates the training loop for the graph
distillation Model using eigenbasis matching. The Trainer receives a Model instance,
a GraphData instance (from dataset_loader.py), and a configuration dictionary (loaded from config.yaml).
It sets up two separate Adam optimizers – one for the synthetic eigenbasis and one for the synthetic
node features – and alternates updates based on the configuration parameters tau1 and tau2.

During each epoch, the Trainer:
  1. Determines the update phase (eigenbasis update if (epoch % (tau1+tau2)) < tau1,
     or synthetic feature update otherwise).
  2. Zeros the appropriate optimizer’s gradients.
  3. Performs a forward pass and computes the total loss (weighted sum of Lₑ, L_d, and Lₒ).
  4. Backpropagates and performs an optimizer step.
  5. Logs intermediate loss components and total variation (TV) of the synthetic graph.
  
After training, the Trainer calls the model’s reconstruct_graph() to assemble the synthetic Laplacian
and adjacency matrix using the learned synthetic eigenbasis and the real graph spectrum. These outputs,
along with training logs, are returned.
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from typing import Any, Dict, Tuple, List

# Trainer class definition
class Trainer:
    def __init__(self, model: Any, data: Any, config: Dict[str, Any]) -> None:
        """
        Initializes the Trainer with a Model instance, a GraphData instance, and a configuration dictionary.
        
        Args:
            model (Model): The graph distillation model (from model.py) with synthetic parameters.
            data (GraphData): An instance containing the real graph data and its eigen-decomposition.
            config (Dict[str, Any]): Configuration dictionary loaded from config.yaml.
        """
        self.model = model
        self.data = data
        self.config = config

        # Extract training hyperparameters from config with defaults
        training_cfg: Dict[str, Any] = config.get("training", {})
        self.epochs: int = int(training_cfg.get("epochs", 1500))
        self.lr_feat: float = float(training_cfg.get("learning_rate_feat", 1e-05))
        self.lr_eigenvecs: float = float(training_cfg.get("learning_rate_eigenvecs", 0.01))
        self.tau1: int = int(training_cfg.get("tau1", 0))
        self.tau2: int = int(training_cfg.get("tau2", 5))
        # Loss weights (alpha, beta, gamma) are used inside Model.compute_loss, so no extra handling is needed here.

        # Set up optimizers.
        # Optimizer for synthetic eigenbasis (U_syn)
        self.optimizer_eigen = optim.Adam([self.model.U_syn], lr=self.lr_eigenvecs)
        # Optimizer for synthetic node features (X_syn)
        self.optimizer_feat = optim.Adam([self.model.X_syn], lr=self.lr_feat)

        # Compute the update period.
        self.period: int = self.tau1 + self.tau2 if (self.tau1 + self.tau2) > 0 else 1

        # Prepare real eigenbasis for loss computation.
        # Model expects real_eigen with shape [N_syn, K_total]. Our dataset returns eigenvectors of shape [N, K_total].
        # We down-sample by taking the first N_syn rows.
        eigenvectors_np: np.ndarray = self.data.eigenvectors  # shape: (num_real_nodes, K_total)
        N_syn: int = self.model.N_syn  # synthetic node count computed in model.__init__
        # Ensure we take at most available rows.
        if eigenvectors_np.shape[0] < N_syn:
            real_eigen_np = eigenvectors_np
        else:
            real_eigen_np = eigenvectors_np[:N_syn, :]
        self.real_eigen: Tensor = torch.tensor(real_eigen_np, dtype=torch.float32)

        # Prepare real eigenvalues for reconstruction
        eigenvalues_np: np.ndarray = self.data.eigenvalues  # shape: (K_total,)
        if eigenvalues_np.shape[0] < N_syn:
            real_eigvals_np = eigenvalues_np
        else:
            real_eigvals_np = eigenvalues_np[:N_syn]
        self.real_eigvals: Tensor = torch.tensor(real_eigvals_np, dtype=torch.float32)

        # Compute real category-level representations.
        # For each unique class in real labels, compute mean feature vector using the entire real node_features.
        self.real_cat_repr: Tensor = self._compute_real_category_repr(self.data.node_features, self.data.labels)
        
        # Prepare synthetic labels.
        # For simplicity, assign synthetic labels as the first N_syn labels from real graph.
        real_labels_np: np.ndarray = self.data.labels
        if real_labels_np.shape[0] < N_syn:
            synthetic_labels_np = real_labels_np
        else:
            synthetic_labels_np = real_labels_np[:N_syn]
        self.synthetic_labels: Tensor = torch.tensor(synthetic_labels_np, dtype=torch.long)

        # Logging containers
        self.loss_history: List[float] = []
        self.L_e_history: List[float] = []
        self.L_d_history: List[float] = []
        self.L_o_history: List[float] = []
        self.tv_history: List[float] = []

    def _compute_real_category_repr(self, features: np.ndarray, labels: np.ndarray) -> Tensor:
        """
        Computes the category-level representations for the real graph as the mean feature vector for each class.
        
        Args:
            features (np.ndarray): Real node feature matrix of shape (N, d).
            labels (np.ndarray): Real node labels of shape (N,).
        
        Returns:
            Tensor: A tensor of shape (num_classes, d) where each row represents the mean feature of a class.
        """
        unique_classes = np.unique(labels)
        cat_reprs = []
        for cls in unique_classes:
            indices = np.where(labels == cls)[0]
            if indices.size == 0:
                continue
            cat_features = features[indices]
            cat_mean = np.mean(cat_features, axis=0)
            cat_reprs.append(cat_mean)
        if len(cat_reprs) == 0:
            # Fallback to zero vector if no class found.
            d = features.shape[1]
            cat_reprs = [np.zeros(d, dtype=np.float32)]
        cat_reprs_np = np.stack(cat_reprs, axis=0)
        return torch.tensor(cat_reprs_np, dtype=torch.float32)

    def _compute_loss_components(self) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Re-computes the loss components (L_e, L_d, L_o) separately for logging purposes.
        These computations mimic the ones in Model.compute_loss.
        
        Returns:
            Tuple[Tensor, Tensor, Tensor]: Eigenbasis loss (L_e), Discrimination loss (L_d), Orthogonality loss (L_o)
        """
        # Forward pass: Use current synthetic parameters.
        X_syn, U_syn = self.model.forward(None)

        # Eigenbasis Matching Loss L_e:
        # Using provided real_eigen from Trainer (shape: [N_syn, K_total])
        P_real = torch.matmul(self.real_eigen, self.real_eigen.transpose(0, 1))
        P_syn = torch.matmul(U_syn, U_syn.transpose(0, 1))
        L_e = torch.norm(P_real - P_syn, p='fro') ** 2

        # Discrimination Constraint Loss L_d:
        unique_classes = torch.unique(self.synthetic_labels)
        synthetic_cat_list = []
        real_cat_list = []
        for cls in unique_classes:
            indices = (self.synthetic_labels == cls).nonzero(as_tuple=True)[0]
            if indices.numel() == 0:
                continue
            cat_mean = torch.mean(X_syn[indices], dim=0)
            synthetic_cat_list.append(cat_mean)
            # Use the class index from real_cat_repr (assumes ordering is aligned)
            real_cat_list.append(self.real_cat_repr[int(cls.item())])
        if len(synthetic_cat_list) > 0:
            synthetic_cat_repr = torch.stack(synthetic_cat_list, dim=0)
            real_cat_repr_subset = torch.stack(real_cat_list, dim=0)
            cos_sim = F.cosine_similarity(synthetic_cat_repr, real_cat_repr_subset, dim=1, eps=1e-8)
            L_d = 1.0 - torch.mean(cos_sim)
        else:
            L_d = torch.tensor(0.0, device=X_syn.device)

        # Orthogonality Loss L_o:
        U_transpose_U = torch.matmul(U_syn.transpose(0, 1), U_syn)
        identity = torch.eye(self.model.K_total, device=U_syn.device, dtype=U_syn.dtype)
        L_o = torch.norm(U_transpose_U - identity, p='fro') ** 2

        return L_e, L_d, L_o

    def _compute_total_variation(self, L_syn: Tensor) -> float:
        """
        Computes the total variation (TV) of the synthetic graph's node features.
        TV is computed as: TV = trace(X_syn^T L_syn X_syn).
        
        Args:
            L_syn (Tensor): The synthetic Laplacian matrix of shape (N_syn, N_syn).
        
        Returns:
            float: The computed total variation.
        """
        X_syn = self.model.X_syn  # shape: (N_syn, d)
        # Compute X_syn^T L_syn X_syn
        tv_matrix = torch.matmul(torch.matmul(X_syn.transpose(0, 1), L_syn), X_syn)
        tv = torch.trace(tv_matrix)
        return tv.item()

    def train(self) -> Dict[str, Any]:
        """
        Executes the training loop for the specified number of epochs.
        Alternates updates between synthetic eigenbasis and synthetic node features depending on the configuration.
        Logs loss components and total variation at regular intervals.
        
        Returns:
            Dict[str, Any]: A dictionary with final synthetic graph structures and training logs:
                - "synthetic_laplacian": Reconstructed synthetic Laplacian matrix.
                - "synthetic_adjacency": Reconstructed synthetic adjacency matrix.
                - "final_loss": Final training loss.
                - "loss_history": List of total loss values per logging interval.
                - "L_e_history": List of eigenbasis loss values per logging interval.
                - "L_d_history": List of discrimination loss values per logging interval.
                - "L_o_history": List of orthogonality loss values per logging interval.
                - "tv_history": List of total variation values per logging interval.
        """
        # For logging now and then.
        log_interval: int = 50  # Log every 50 epochs.
        
        for epoch in range(1, self.epochs + 1):
            # Determine update phase
            # If epoch modulo period < tau1, then update eigenbasis; otherwise, update node features.
            if (epoch % self.period) < self.tau1:
                update_mode = "eigenbasis"
                optimizer = self.optimizer_eigen
            else:
                update_mode = "features"
                optimizer = self.optimizer_feat

            # Zero the gradients for the selected optimizer.
            optimizer.zero_grad()

            # Compute total loss using model.compute_loss.
            total_loss: Tensor = self.model.compute_loss(
                real_eigen=self.real_eigen, 
                real_cat_repr=self.real_cat_repr, 
                synthetic_labels=self.synthetic_labels
            )

            # Backpropagation.
            total_loss.backward()
            # Optimizer step.
            optimizer.step()

            # Logging losses every epoch (or every log_interval epochs).
            with torch.no_grad():
                # Recompute individual loss components for logging.
                L_e_val, L_d_val, L_o_val = self._compute_loss_components()
                # Reconstruct synthetic graph to compute total variation TV.
                L_syn, _ = self.model.reconstruct_graph(self.real_eigvals)
                tv_val: float = self._compute_total_variation(L_syn)

            # Append values to logging lists.
            self.loss_history.append(total_loss.item())
            self.L_e_history.append(L_e_val.item())
            self.L_d_history.append(L_d_val.item())
            self.L_o_history.append(L_o_val.item())
            self.tv_history.append(tv_val)

            # Print logs at regular intervals.
            if epoch % log_interval == 0 or epoch == 1 or epoch == self.epochs:
                print(f"Epoch [{epoch}/{self.epochs}] | Update Mode: {update_mode} | Total Loss: {total_loss.item():.6f} "
                      f"| L_e: {L_e_val.item():.6f} | L_d: {L_d_val.item():.6f} | L_o: {L_o_val.item():.6f} | TV: {tv_val:.6f}")

        # After training, reconstruct the synthetic graph.
        synthetic_L, synthetic_A = self.model.reconstruct_graph(self.real_eigvals)

        final_logs = {
            "synthetic_laplacian": synthetic_L.detach().cpu().numpy(),
            "synthetic_adjacency": synthetic_A.detach().cpu().numpy(),
            "final_loss": self.loss_history[-1] if self.loss_history else None,
            "loss_history": self.loss_history,
            "L_e_history": self.L_e_history,
            "L_d_history": self.L_d_history,
            "L_o_history": self.L_o_history,
            "tv_history": self.tv_history
        }
        print("Training completed.")
        return final_logs
