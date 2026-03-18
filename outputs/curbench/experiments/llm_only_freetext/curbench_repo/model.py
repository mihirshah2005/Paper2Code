"""model.py

This module defines the ModelManager class which is responsible for building and returning
a PyTorch nn.Module instance that serves as the backbone model for a given experiment.
It supports multiple domains (CV, NLP, Graph) and model types (e.g., LeNet, ResNet-18,
ViT for CV; LSTM, BERT, GPT2 for NLP; GCN, GAT, GIN for Graph). The model is adjusted
according to dataset-specific properties provided via the dataset_info dictionary.
"""

import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from transformers import BertForSequenceClassification, GPT2ForSequenceClassification

from torch_geometric.nn import GCNConv, GATConv, GINConv, global_mean_pool

# Configure module level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# ======================
# CV Models
# ======================

class LeNet(nn.Module):
    """LeNet - A basic convolutional neural network model for CV tasks."""
    def __init__(self, in_channels: int = 3, num_classes: int = 10) -> None:
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # For a 32x32 input image: after conv1 (28x28) then pool (14x14), after conv2 (10x10) then pool (5x5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class VisionTransformer(nn.Module):
    """
    VisionTransformer - A minimal implementation of a Vision Transformer for CV tasks.
    It divides the input image into patches, projects them, adds a class token and positional embeddings,
    and passes the sequence through a Transformer encoder.
    """
    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        embed_dim: int = 64,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1
    ) -> None:
        super(VisionTransformer, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        if image_size % patch_size != 0:
            raise ValueError("Image size must be divisible by patch size.")
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size
        self.embed_dim = embed_dim

        # Linear projection of flattened patches
        self.proj = nn.Linear(self.patch_dim, embed_dim)

        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Positional embeddings for all patches + class token
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Classification head
        self.fc = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, H, W)
        B = x.size(0)
        # Extract patches using unfold
        patches = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size)
        # patches shape: (B, patch_dim, num_patches)
        patches = patches.transpose(1, 2)  # (B, num_patches, patch_dim)
        embeddings = self.proj(patches)    # (B, num_patches, embed_dim)

        # Prepend class token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, embeddings), dim=1)   # (B, num_patches + 1, embed_dim)
        x = x + self.pos_embed
        x = self.dropout(x)

        # Transformer expects input shape: (sequence_length, batch_size, embed_dim)
        x = x.transpose(0, 1)
        x = self.transformer(x)
        # Use the output corresponding to the class token
        x = x[0]  # (B, embed_dim)
        x = self.fc(x)  # (B, num_classes)
        return x


# ======================
# NLP Models
# ======================

class LSTMClassifier(nn.Module):
    """
    LSTMClassifier - An LSTM-based classifier for NLP tasks.
    Consists of an embedding layer, an LSTM module, and a final linear layer for classification.
    """
    def __init__(
        self,
        vocab_size: int = 10000,
        embedding_dim: int = 300,
        hidden_dim: int = 128,
        num_layers: int = 1,
        num_classes: int = 2,
        dropout: float = 0.5
    ) -> None:
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, seq_length)
        embedded = self.dropout(self.embedding(x))  # (B, seq_length, embedding_dim)
        output, (hidden, _) = self.lstm(embedded)
        # Use the last layer's hidden state for classification
        last_hidden = hidden[-1]  # (B, hidden_dim)
        logits = self.fc(last_hidden)  # (B, num_classes)
        return logits


# ======================
# Graph Models
# ======================

class GraphGCN(nn.Module):
    """
    GraphGCN - A Graph Convolutional Network for graph tasks.
    Uses two GCNConv layers followed by a global mean pooling and a final fully-connected layer.
    """
    def __init__(self, num_features: int, hidden_dim: int = 64, num_classes: int = 2) -> None:
        super(GraphGCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, data: any) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.fc(x)


class GraphGAT(nn.Module):
    """
    GraphGAT - A Graph Attention Network for graph tasks.
    Utilizes GATConv layers and a global pooling mechanism.
    """
    def __init__(self, num_features: int, hidden_dim: int = 64, num_classes: int = 2, heads: int = 8) -> None:
        super(GraphGAT, self).__init__()
        self.conv1 = GATConv(num_features, hidden_dim, heads=heads, concat=True)
        # When concatenating multiple heads, the output dimension is hidden_dim * heads.
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, data: any) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.fc(x)


class GraphGIN(nn.Module):
    """
    GraphGIN - A Graph Isomorphism Network for graph tasks.
    Uses GINConv layers (with internal MLPs) and global mean pooling, followed by a final classification layer.
    """
    def __init__(self, num_features: int, hidden_dim: int = 64, num_classes: int = 2) -> None:
        super(GraphGIN, self).__init__()
        nn1 = nn.Sequential(nn.Linear(num_features, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.conv1 = GINConv(nn1)
        nn2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.conv2 = GINConv(nn2)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, data: any) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.fc(x)


# ======================
# ModelManager Class
# ======================

class ModelManager:
    """
    ModelManager provides a unified interface to build backbone models for various domains and tasks.
    It uses the configuration dictionary to verify supported model types and returns a fully constructed
    nn.Module adjusted to the dataset specifications.
    """
    def __init__(self, config: dict) -> None:
        """
        Initialize ModelManager with the provided configuration.
        :param config: Configuration dictionary loaded from config.yaml.
        """
        self.config = config
        # Supported models as specified in the configuration
        self.supported_models = {
            "cv": ["LeNet", "ResNet-18", "ViT"],
            "nlp": ["LSTM", "BERT", "GPT2"],
            "graph": ["GCN", "GAT", "GIN"]
        }
        logger.info("ModelManager initialized with configuration.")

    def build_model(self, domain: str, model_name: str, dataset_info: dict) -> nn.Module:
        """
        Build and return a backbone model for the given domain and model name.
        :param domain: A string indicating the domain ("cv", "nlp", or "graph").
        :param model_name: The model to instantiate (e.g., "LeNet", "ResNet-18", "ViT", "LSTM", "BERT", "GPT2", "GCN", "GAT", or "GIN").
        :param dataset_info: Dictionary of dataset-specific properties (e.g., number of classes, image channels).
        :return: An instance of nn.Module representing the model.
        :raises ValueError: If an unsupported domain or model_name is provided.
        """
        domain = domain.lower()
        model_name = model_name.strip()
        if domain not in self.supported_models:
            raise ValueError(
                f"Unsupported domain: {domain}. Supported domains are 'cv', 'nlp', and 'graph'."
            )
        if model_name not in self.supported_models[domain]:
            raise ValueError(
                f"Unsupported model: {model_name} for domain: {domain}. Supported models are: {self.supported_models[domain]}."
            )
        logger.info(f"Building model '{model_name}' for domain '{domain}'.")
        if domain == "cv":
            return self._build_cv_model(model_name, dataset_info)
        elif domain == "nlp":
            return self._build_nlp_model(model_name, dataset_info)
        elif domain == "graph":
            return self._build_graph_model(model_name, dataset_info)
        else:
            raise NotImplementedError(f"Model building for domain {domain} is not implemented.")

    def _build_cv_model(self, model_name: str, dataset_info: dict) -> nn.Module:
        """
        Build a model for the CV domain.
        :param model_name: Model name (e.g., "LeNet", "ResNet-18", "ViT").
        :param dataset_info: Contains "num_classes", and optionally "in_channels", "image_size", etc.
        :return: A cv model instance.
        """
        num_classes = dataset_info.get("num_classes", 10)
        in_channels = dataset_info.get("in_channels", 3)
        if model_name == "LeNet":
            return LeNet(in_channels=in_channels, num_classes=num_classes)
        elif model_name == "ResNet-18":
            model = models.resnet18(pretrained=False)
            # Adjust first convolution if input channels differ from 3
            if in_channels != 3:
                model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
            return model
        elif model_name == "ViT":
            image_size = dataset_info.get("image_size", 32)
            patch_size = dataset_info.get("patch_size", 4)
            embed_dim = dataset_info.get("embed_dim", 64)
            num_layers = dataset_info.get("num_layers", 6)
            num_heads = dataset_info.get("num_heads", 8)
            dropout = dataset_info.get("dropout", 0.1)
            return VisionTransformer(
                image_size=image_size,
                patch_size=patch_size,
                in_channels=in_channels,
                num_classes=num_classes,
                embed_dim=embed_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unsupported CV model: {model_name}")

    def _build_nlp_model(self, model_name: str, dataset_info: dict) -> nn.Module:
        """
        Build a model for the NLP domain.
        :param model_name: Model name ("LSTM", "BERT", or "GPT2").
        :param dataset_info: Contains "num_classes", and for LSTM possibly "vocab_size", "embedding_dim", etc.
        :return: An NLP model instance.
        """
        num_classes = dataset_info.get("num_classes", 2)
        if model_name == "LSTM":
            vocab_size = dataset_info.get("vocab_size", 10000)
            embedding_dim = dataset_info.get("embedding_dim", 300)
            hidden_dim = dataset_info.get("hidden_dim", 128)
            num_layers = dataset_info.get("num_layers", 1)
            dropout = dataset_info.get("dropout", 0.5)
            return LSTMClassifier(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_classes=num_classes,
                dropout=dropout
            )
        elif model_name == "BERT":
            return BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_classes)
        elif model_name == "GPT2":
            model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=num_classes)
            # GPT2 does not have a pad_token by default; set it to eos_token_id.
            model.config.pad_token_id = model.config.eos_token_id
            return model
        else:
            raise ValueError(f"Unsupported NLP model: {model_name}. Supported models: ['LSTM', 'BERT', 'GPT2'].")

    def _build_graph_model(self, model_name: str, dataset_info: dict) -> nn.Module:
        """
        Build a model for the Graph domain.
        :param model_name: Model name ("GCN", "GAT", or "GIN").
        :param dataset_info: Contains "num_classes" and "num_features"; may also include "hidden_dim", "heads", etc.
        :return: A graph model instance.
        """
        num_classes = dataset_info.get("num_classes", 2)
        num_features = dataset_info.get("num_features", 16)
        if model_name == "GCN":
            hidden_dim = dataset_info.get("hidden_dim", 64)
            return GraphGCN(num_features=num_features, hidden_dim=hidden_dim, num_classes=num_classes)
        elif model_name == "GAT":
            hidden_dim = dataset_info.get("hidden_dim", 64)
            heads = dataset_info.get("heads", 8)
            return GraphGAT(num_features=num_features, hidden_dim=hidden_dim, num_classes=num_classes, heads=heads)
        elif model_name == "GIN":
            hidden_dim = dataset_info.get("hidden_dim", 64)
            return GraphGIN(num_features=num_features, hidden_dim=hidden_dim, num_classes=num_classes)
        else:
            raise ValueError(f"Unsupported Graph model: {model_name}. Supported models: ['GCN', 'GAT', 'GIN'].")
