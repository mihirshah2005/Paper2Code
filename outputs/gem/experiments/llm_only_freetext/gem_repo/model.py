"""
model.py

This module implements the Model class that performs graph distillation via eigenbasis matching.
It defines learnable synthetic parameters, computes the composite loss (eigenbasis matching loss Lₑ,
discrimination constraint loss L_d, and orthogonality regularization loss L_o), and reconstructs
the synthetic graph (Laplacian and adjacency) using the learned synthetic eigenbasis and real spectrum.

The design assumes that the Model is initialized with a configuration dictionary (from config.yaml)
and a GraphData object obtained from the DatasetLoader. The GraphData object provides the real graph's
node features and eigen-decomposition. Synthetic node features and an eigenbasis (U′_K) are initialized
as learnable parameters. The forward() method returns the current synthetic parameters, while compute_loss()
calculates the total loss used for training. Methods update_eigenbasis() and update_features() provide
manual parameter updates (if not using the optimizer). Finally, reconstruct_graph() rebuilds the synthetic
graph using the real eigenvalues.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Any, Dict, Tuple
import math

class Model(nn.Module):
    def __init__(self, config: Dict[str, Any], graph_data: Any) -> None:
        """
        Initializes the Model with the given configuration and graph data.

        Args:
            config (Dict[str, Any]): Configuration dictionary (from config.yaml).
            graph_data (GraphData): Graph data object that includes node_features and real eigenvectors.
        
        The synthetic graph dimensions are computed based on the compression ratio and the real graph's node count.
        Synthetic node features (X′, shape: [N_syn, d]) and synthetic eigenbasis (U′_K, shape: [N_syn, K_total])
        are created as learnable parameters.
        """
        super(Model, self).__init__()
        
        # Retrieve training hyperparameters.
        training_config: Dict[str, Any] = config.get("training", {})
        self.lr_feat: float = float(training_config.get("learning_rate_feat", 1e-05))
        self.lr_eigenvecs: float = float(training_config.get("learning_rate_eigenvecs", 0.01))
        loss_weights: Dict[str, Any] = training_config.get("loss_weights", {"alpha": 1.0, "beta": 1.0, "gamma": 1.0})
        self.alpha: float = float(loss_weights.get("alpha", 1.0))
        self.beta: float = float(loss_weights.get("beta", 1.0))
        self.gamma: float = float(loss_weights.get("gamma", 1.0))
        
        # Retrieve synthetic graph parameters.
        synth_config: Dict[str, Any] = config.get("synth_graph", {})
        # compression_ratio is defined as a percentage (e.g., 0.15 means 0.15%)
        comp_ratio: float = float(synth_config.get("compression_ratio", 0.15))
        r_k: float = float(synth_config.get("r_k", 0.9))
        
        # Determine synthetic graph dimensions.
        # Feature dimension (d) from the real graph node features.
        real_features: Tensor = torch.tensor(graph_data.node_features)
        self.feature_dim: int = real_features.size(1)
        # Number of nodes in the real graph.
        num_real_nodes: int = graph_data.node_features.shape[0]
        # Compute synthetic node count (N_syn). Interpreting compression_ratio as a percentage.
        N_syn: int = max(1, int(num_real_nodes * (comp_ratio / 100.0)))
        self.N_syn: int = N_syn
        
        # For eigenbasis matching, the total number K of eigenpairs to match = N_syn.
        self.K_total: int = N_syn  # Total number of eigenvectors to match.
        self.K1: int = int(r_k * N_syn)  # Low-frequency (global) component.
        if self.K1 == 0 and N_syn > 0:
            self.K1 = 1
        self.K2: int = N_syn - self.K1  # High-frequency (local) component.
        
        # Initialize synthetic node features X′ ∈ ℝ^(N_syn x d) as a learnable parameter.
        self.X_syn: nn.Parameter = nn.Parameter(torch.randn(N_syn, self.feature_dim, dtype=torch.float32))
        
        # Initialize synthetic eigenbasis U′_K ∈ ℝ^(N_syn x K_total) as a learnable parameter.
        # Use random initialization and then perform orthogonal initialization.
        self.U_syn: nn.Parameter = nn.Parameter(torch.randn(N_syn, self.K_total, dtype=torch.float32))
        self._init_orthogonal(self.U_syn)
    
    def _init_orthogonal(self, param: nn.Parameter) -> None:
        """
        Initializes the parameter with an orthogonal matrix.
        If the number of rows is greater than or equal to the number of columns,
        performs standard orthogonal initialization.
        """
        if param.dim() < 2:
            raise ValueError("Only parameters with 2 or more dimensions are supported for orthogonal initialization")
        # Use torch.nn.init.orthogonal_ for a 2D tensor.
        nn.init.orthogonal_(param)
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass that returns the current synthetic node features X′ and synthetic eigenbasis U′_K.
        Although an input tensor x is accepted for API compatibility, it is intentionally unused.

        Args:
            x (Tensor): Input tensor (unused).

        Returns:
            Tuple[Tensor, Tensor]: (X_syn, U_syn)
        """
        # x is intentionally ignored as the synthesis process directly learns its parameters.
        return self.X_syn, self.U_syn

    def compute_loss(
        self,
        real_eigen: Tensor,
        real_cat_repr: Tensor,
        synthetic_labels: Tensor
    ) -> Tensor:
        """
        Computes the total loss as a weighted sum of:
         - Eigenbasis Matching Loss (Lₑ): Aligns the projection matrices of the real eigenbasis and synthetic eigenbasis.
         - Discrimination Constraint Loss (L_d): Enforces similarity between real and synthetic category-level representations.
         - Orthogonality Regularization Loss (L_o): Ensures U_syn remains (approximately) an orthonormal matrix.

        Args:
            real_eigen (Tensor): Preprocessed real eigenbasis (truncated) with shape [N_syn, K_total].
                                 It is assumed that the dimension matches that of U_syn.
            real_cat_repr (Tensor): Real category-level representations with shape [num_classes, d].
                                    The row index corresponds to the class label.
            synthetic_labels (Tensor): Synthetic node labels with shape [N_syn], values as integers.

        Returns:
            Tensor: The scalar total loss.
        """
        # --- Eigenbasis Matching Loss (Lₑ) ---
        # Compute projection matrices: P_real = real_eigen @ real_eigen^T, P_syn = U_syn @ U_syn^T.
        P_real: Tensor = torch.matmul(real_eigen, real_eigen.transpose(0, 1))
        P_syn: Tensor = torch.matmul(self.U_syn, self.U_syn.transpose(0, 1))
        L_e: Tensor = torch.norm(P_real - P_syn, p='fro') ** 2

        # --- Discrimination Constraint Loss (L_d) ---
        # Derive synthetic category-level representations (H′) by averaging synthetic features per class.
        unique_classes: Tensor = torch.unique(synthetic_labels)
        synthetic_cat_list = []
        real_cat_list = []
        for cls in unique_classes:
            indices: Tensor = (synthetic_labels == cls).nonzero(as_tuple=True)[0]
            if indices.numel() == 0:
                continue
            # Mean synthetic feature for class 'cls'.
            cat_mean: Tensor = torch.mean(self.X_syn[indices], dim=0)
            synthetic_cat_list.append(cat_mean)
            # Assume that real_cat_repr rows are aligned with class labels.
            # Using int(cls) as index.
            real_cat_list.append(real_cat_repr[int(cls)])
        if len(synthetic_cat_list) > 0:
            synthetic_cat_repr: Tensor = torch.stack(synthetic_cat_list, dim=0)
            real_cat_repr_subset: Tensor = torch.stack(real_cat_list, dim=0)
            # Compute cosine similarity for each corresponding class.
            cos_sim: Tensor = F.cosine_similarity(synthetic_cat_repr, real_cat_repr_subset, dim=1, eps=1e-8)
            L_d: Tensor = 1.0 - torch.mean(cos_sim)
        else:
            L_d: Tensor = torch.tensor(0.0, device=self.X_syn.device)

        # --- Orthogonality Regularization Loss (L_o) ---
        # Compute the Gram matrix of U_syn: G = U_syn^T U_syn and enforce G ≈ I.
        U_transpose_U: Tensor = torch.matmul(self.U_syn.transpose(0, 1), self.U_syn)
        identity: Tensor = torch.eye(self.K_total, device=U_transpose_U.device, dtype=U_transpose_U.dtype)
        L_o: Tensor = torch.norm(U_transpose_U - identity, p='fro') ** 2

        # Combine the losses.
        total_loss: Tensor = self.alpha * L_e + self.beta * L_d + self.gamma * L_o
        return total_loss

    def update_eigenbasis(self, gradients: Tensor) -> None:
        """
        Manually updates the synthetic eigenbasis parameter U_syn using the provided gradients
        and the learning rate for eigenvectors.
        Note: In practice, optimizer.step() is preferred; this method is provided to conform
        to the design interface.

        Args:
            gradients (Tensor): Gradients corresponding to U_syn.
        """
        self.U_syn.data = self.U_syn.data - self.lr_eigenvecs * gradients

    def update_features(self, gradients: Tensor) -> None:
        """
        Manually updates the synthetic node features parameter X_syn using the provided gradients
        and the learning rate for features.

        Args:
            gradients (Tensor): Gradients corresponding to X_syn.
        """
        self.X_syn.data = self.X_syn.data - self.lr_feat * gradients

    def reconstruct_graph(self, real_eigvals: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Reconstructs the synthetic graph structure using the learned synthetic eigenbasis and
        the real graph's eigenvalues.
        
        The synthetic Laplacian L_syn is computed as:
            L_syn = U_syn @ diag(real_eigvals) @ U_syn^T,
        and the synthetic adjacency matrix A_syn is computed as:
            A_syn = U_syn @ diag(1 - real_eigvals) @ U_syn^T.

        Args:
            real_eigvals (Tensor): A 1D tensor of length K_total representing the selected real eigenvalues.

        Returns:
            Tuple[Tensor, Tensor]: A tuple (L_syn, A_syn) representing the synthetic Laplacian and adjacency matrix.
        """
        if real_eigvals.dim() != 1 or real_eigvals.size(0) != self.K_total:
            raise ValueError("real_eigvals must be a 1D tensor with length equal to K_total (%d)" % self.K_total)
        diag_eig: Tensor = torch.diag(real_eigvals)
        L_syn: Tensor = torch.matmul(torch.matmul(self.U_syn, diag_eig), self.U_syn.transpose(0, 1))
        diag_adj: Tensor = torch.diag(1.0 - real_eigvals)
        A_syn: Tensor = torch.matmul(torch.matmul(self.U_syn, diag_adj), self.U_syn.transpose(0, 1))
        return L_syn, A_syn
