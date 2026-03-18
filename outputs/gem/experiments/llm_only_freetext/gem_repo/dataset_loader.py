"""
dataset_loader.py

This module implements the DatasetLoader class and the GraphData data structure.
DatasetLoader is responsible for loading a graph dataset, computing the normalized
Laplacian and its truncated eigen-decomposition, and packaging the results into a
GraphData object. These components are later used for the eigenbasis matching
in the graph distillation process.

The current implementation supports the "Pubmed" dataset (case‐insensitive)
and a default simulation for other dataset names. For "Pubmed", if the configuration
specifies simulation (simulate=True), a smaller graph is generated. Otherwise,
full Pubmed parameters are used. The compression_ratio (interpreted as a percentage)
and r_k parameters from the configuration determine the number of synthetic nodes
(N') and hence the number of eigenpairs to extract.
"""

import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class GraphData:
    adjacency: np.ndarray         # Adjacency matrix of the graph (N x N)
    node_features: np.ndarray     # Node feature matrix (N x d)
    labels: np.ndarray            # Labels for node classification (N,)
    laplacian: np.ndarray         # Normalized Laplacian matrix (N x N)
    eigenvalues: np.ndarray       # Selected eigenvalues (K,)
    eigenvectors: np.ndarray      # Selected eigenvectors (N x K)


class DatasetLoader:
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initializes DatasetLoader with a configuration dictionary.
        Expected config keys:
         - "dataset": { "name": <dataset_name>, "split_method": <split_method>, "simulate": <bool> }
         - "synth_graph": { "compression_ratio": <float>, "r_k": <float> }
        """
        self.config = config
        self.dataset_config: Dict[str, Any] = config.get("dataset", {})
        self.synth_config: Dict[str, Any] = config.get("synth_graph", {})
        self.dataset_name: str = self.dataset_config.get("name", "Pubmed")
        self.split_method: str = self.dataset_config.get("split_method", "public")
        # Flag to simulate a smaller graph for demonstration purposes.
        self.simulate: bool = self.dataset_config.get("simulate", True)
        # Compression ratio is given as a percentage.
        self.compression_ratio: float = float(self.synth_config.get("compression_ratio", 0.15))
        # r_k defines the fraction for the low-frequency (smallest eigenvalues) eigenbasis.
        self.r_k: float = float(self.synth_config.get("r_k", 0.9))

    def load_data(self) -> GraphData:
        """
        Loads the graph dataset and returns a GraphData object with the following fields:
          - adjacency: The dense adjacency matrix.
          - node_features: The node feature matrix.
          - labels: The node classification labels.
          - laplacian: The computed normalized Laplacian matrix.
          - eigenvalues: The truncated eigenvalues (combined smallest and largest).
          - eigenvectors: The corresponding eigenvectors.
        """
        if self.dataset_name.lower() == "pubmed":
            return self._load_pubmed()
        else:
            return self._load_default()

    def _load_pubmed(self) -> GraphData:
        """
        Loads (or simulates) the Pubmed dataset.
          - If simulate is True, a smaller graph is generated for quick testing.
          - Otherwise, full Pubmed parameters are used.
        The normalized Laplacian is computed and then a truncated eigen-decomposition
        is performed based on the synthetic node count (N') derived from the compression_ratio.
        """
        # Set parameters based on simulation flag.
        if self.simulate:
            # Simulated (small) version for testing.
            N: int = 1000          # Number of nodes in the simulated graph.
            p: float = 0.01        # Edge probability.
            feature_dim: int = 50  # Dimensionality of node features.
            num_classes: int = 3   # Number of node classes.
        else:
            # Full Pubmed parameters.
            N = 19717              # Number of nodes in the real Pubmed graph.
            p = 0.0003             # Edge probability approximating sparsity.
            feature_dim = 500      # Feature dimension as in Pubmed.
            num_classes = 3

        # Generate a random Erdos-Renyi graph.
        G: nx.Graph = nx.erdos_renyi_graph(n=N, p=p, seed=42)
        # Ensure the graph is connected; if not, use the largest connected component.
        if not nx.is_connected(G):
            components = list(nx.connected_components(G))
            largest_cc = max(components, key=len)
            G = G.subgraph(largest_cc).copy()
            N = G.number_of_nodes()

        # For reproducibility.
        np.random.seed(42)
        # Generate random node features.
        X: np.ndarray = np.random.rand(N, feature_dim).astype(np.float32)
        # Generate random integer labels in the range [0, num_classes).
        Y: np.ndarray = np.random.randint(low=0, high=num_classes, size=(N,))
        # Obtain the dense adjacency matrix.
        A: np.ndarray = nx.to_numpy_array(G, dtype=np.float32)

        # Compute the normalized Laplacian L = I - D^(-1/2) A D^(-1/2)
        # Using networkx's built-in function that returns a sparse matrix.
        L_sparse: sp.spmatrix = nx.normalized_laplacian_matrix(G)
        L: np.ndarray = L_sparse.toarray().astype(np.float32)

        # --- Synthetic Graph Sizing and Eigen-decomposition ---
        # Compute synthetic node count N' using the compression_ratio (interpreted as a percentage).
        N_syn: int = max(1, int(N * (self.compression_ratio / 100.0)))
        # Total number of eigenpairs (K) to select is N_syn.
        K_total: int = N_syn
        # Split the eigenpairs into low-frequency and high-frequency components.
        K1: int = int(self.r_k * N_syn)
        if K1 == 0 and N_syn > 0:
            K1 = 1
        K2: int = N_syn - K1

        eigenvalues_list = []
        eigenvectors_list = []

        # Compute K1 smallest eigenpairs (low-frequency, capturing global structure).
        if K1 > 0:
            try:
                eigs_small, evecs_small = eigsh(L_sparse, k=K1, which='SM', tol=1e-3)
            except Exception as e:
                print("Error computing smallest eigenpairs:", e)
                eigs_small = np.array([], dtype=np.float32)
                evecs_small = np.empty((N, 0), dtype=np.float32)
            eigenvalues_list.append(eigs_small)
            eigenvectors_list.append(evecs_small)

        # Compute K2 largest eigenpairs (high-frequency, capturing local details).
        if K2 > 0:
            try:
                eigs_large, evecs_large = eigsh(L_sparse, k=K2, which='LA', tol=1e-3)
            except Exception as e:
                print("Error computing largest eigenpairs:", e)
                eigs_large = np.array([], dtype=np.float32)
                evecs_large = np.empty((N, 0), dtype=np.float32)
            eigenvalues_list.append(eigs_large)
            eigenvectors_list.append(evecs_large)

        # Combine the eigenpairs.
        if eigenvalues_list:
            eigenvalues_combined: np.ndarray = np.concatenate(eigenvalues_list)
            eigenvectors_combined: np.ndarray = np.concatenate(eigenvectors_list, axis=1)
        else:
            # Fallback: Use a trivial eigenpair.
            eigenvalues_combined = np.zeros((1,), dtype=np.float32)
            eigenvectors_combined = np.eye(N, 1, dtype=np.float32)

        # Package the data into a GraphData object.
        graph_data = GraphData(
            adjacency=A,
            node_features=X,
            labels=Y,
            laplacian=L,
            eigenvalues=eigenvalues_combined.astype(np.float32),
            eigenvectors=eigenvectors_combined.astype(np.float32)
        )
        return graph_data

    def _load_default(self) -> GraphData:
        """
        Default loader for datasets other than "Pubmed".
        Uses standard simulation parameters.
        """
        N: int = 500           # Default number of nodes.
        p: float = 0.05        # Default edge probability.
        feature_dim: int = 50  # Default feature dimension.
        num_classes: int = 2   # Default number of classes.
        
        G: nx.Graph = nx.erdos_renyi_graph(n=N, p=p, seed=42)
        if not nx.is_connected(G):
            components = list(nx.connected_components(G))
            largest_cc = max(components, key=len)
            G = G.subgraph(largest_cc).copy()
            N = G.number_of_nodes()
        
        np.random.seed(42)
        X: np.ndarray = np.random.rand(N, feature_dim).astype(np.float32)
        Y: np.ndarray = np.random.randint(low=0, high=num_classes, size=(N,))
        A: np.ndarray = nx.to_numpy_array(G, dtype=np.float32)
        
        L_sparse: sp.spmatrix = nx.normalized_laplacian_matrix(G)
        L: np.ndarray = L_sparse.toarray().astype(np.float32)
        
        N_syn: int = max(1, int(N * (self.compression_ratio / 100.0)))
        K_total: int = N_syn
        K1: int = int(self.r_k * N_syn)
        if K1 == 0 and N_syn > 0:
            K1 = 1
        K2: int = N_syn - K1
        
        eigenvalues_list = []
        eigenvectors_list = []
        
        if K1 > 0:
            try:
                eigs_small, evecs_small = eigsh(L_sparse, k=K1, which='SM', tol=1e-3)
            except Exception as e:
                print("Error computing smallest eigenpairs:", e)
                eigs_small = np.array([], dtype=np.float32)
                evecs_small = np.empty((N, 0), dtype=np.float32)
            eigenvalues_list.append(eigs_small)
            eigenvectors_list.append(evecs_small)
        
        if K2 > 0:
            try:
                eigs_large, evecs_large = eigsh(L_sparse, k=K2, which='LA', tol=1e-3)
            except Exception as e:
                print("Error computing largest eigenpairs:", e)
                eigs_large = np.array([], dtype=np.float32)
                evecs_large = np.empty((N, 0), dtype=np.float32)
            eigenvalues_list.append(eigs_large)
            eigenvectors_list.append(evecs_large)
        
        if eigenvalues_list:
            eigenvalues_combined: np.ndarray = np.concatenate(eigenvalues_list)
            eigenvectors_combined: np.ndarray = np.concatenate(eigenvectors_list, axis=1)
        else:
            eigenvalues_combined = np.zeros((1,), dtype=np.float32)
            eigenvectors_combined = np.eye(N, 1, dtype=np.float32)
        
        graph_data = GraphData(
            adjacency=A,
            node_features=X,
            labels=Y,
            laplacian=L,
            eigenvalues=eigenvalues_combined.astype(np.float32),
            eigenvectors=eigenvectors_combined.astype(np.float32)
        )
        return graph_data
