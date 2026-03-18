"""
trainer.py

This module implements the Trainer class which is responsible for retraining a classifier (ResNet-18)
on a purified dataset. The Trainer class sets up device management, reproducibility, DataLoader construction,
loss function, optimizer, learning rate scheduler, and executes the training loop while logging key metrics.

The training hyperparameters are taken from a configuration dictionary (e.g., parsed from config.yaml).
The Trainer ensures reproducibility by setting random seeds for Python, NumPy, and PyTorch, and by fixing
cuDNN's deterministic behavior.
"""

import os
import random
import logging
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import set_random_seed, init_logger

# Initialize module logger
logger = init_logger()


class Trainer:
    """
    Trainer class for retraining a classifier on the purified dataset.

    Attributes:
        model (torch.nn.Module): The classifier model (e.g., ResNet-18) to be trained.
        train_loader (DataLoader): DataLoader built from the purified training dataset.
        criterion (nn.Module): Loss function used (CrossEntropyLoss).
        optimizer (optim.Optimizer): SGD optimizer for model parameter updates.
        scheduler (optim.lr_scheduler._LRScheduler): Learning rate scheduler (MultiStepLR).
        device (torch.device): Computation device ("cuda" or "cpu").
        epochs (int): Number of training epochs.
    """

    def __init__(self, model: Any, train_data: Any, hyperparams: Dict[str, Any]) -> None:
        """
        Initialize the Trainer with the model, purified training dataset, and hyperparameters.

        Args:
            model (Any): The classifier model (e.g., a ResNet-18 instance).
            train_data (Any): The purified training dataset (torch.utils.data.Dataset).
            hyperparams (Dict[str, Any]): Dictionary of training hyperparameters.
                Expected keys include: "epochs", "learning_rate", "batch_size", "lr_decay_schedule",
                and optionally "seed" and "dataset_name".
        """
        # Set reproducibility seeds.
        seed: int = hyperparams.get("seed", 42)
        set_random_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info("Reproducibility set with seed %d", seed)

        # Device management.
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(self.device)
        self.model: torch.nn.Module = model
        logger.info("Model moved to device: %s", self.device)

        # Determine dataset name; default to "CIFAR-10" if not provided.
        self.dataset_name: str = hyperparams.get("dataset_name", "CIFAR-10")
        # Determine batch size from hyperparameters
        batch_size_dict: Dict[str, int] = hyperparams.get("batch_size", {"CIFAR-10": 128, "ImageNet-100": 64, "ImageNet-Dog": 64})
        self.batch_size: int = batch_size_dict.get(self.dataset_name, 128)
        logger.info("Using dataset '%s' with batch size %d", self.dataset_name, self.batch_size)

        # Create DataLoader for the training dataset.
        # Use a worker_init_fn to ensure each worker gets a proper seed.
        def worker_init_fn(worker_id: int) -> None:
            worker_seed = seed + worker_id
            random.seed(worker_seed)
            np.random.seed(worker_seed)
            torch.manual_seed(worker_seed)

        self.train_loader: DataLoader = DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            worker_init_fn=worker_init_fn
        )
        logger.info("DataLoader initialized with %d samples.", len(train_data))

        # Hyperparameters for training.
        self.epochs: int = hyperparams.get("epochs", 100)
        self.learning_rate: float = hyperparams.get("learning_rate", 0.1)
        logger.info("Training for %d epochs with initial learning rate %.5f", self.epochs, self.learning_rate)

        # Loss function.
        self.criterion: nn.Module = nn.CrossEntropyLoss()

        # Optimizer: using SGD. Default momentum set to 0.9.
        self.optimizer: optim.Optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=0.9
        )

        # Learning rate scheduler.
        lr_decay_schedule: Dict[str, list] = hyperparams.get("lr_decay_schedule", {})
        milestones_list = lr_decay_schedule.get(self.dataset_name, [])
        # Extract milestone epochs from the provided schedule.
        milestones = [int(milestone.get("epoch", 0)) for milestone in milestones_list]
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=milestones,
            gamma=0.1
        )
        logger.info("Learning rate scheduler milestones set to: %s", milestones)

    def train(self) -> torch.nn.Module:
        """
        Execute the training loop over the purified training dataset.

        Returns:
            torch.nn.Module: The trained model.
        """
        self.model.train()

        for epoch in range(1, self.epochs + 1):
            epoch_loss: float = 0.0
            correct: int = 0
            total: int = 0

            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Zero the gradients.
                self.optimizer.zero_grad()

                # Forward pass.
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # Update metrics.
                epoch_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

            # Step the scheduler after each epoch.
            self.scheduler.step()

            avg_loss: float = epoch_loss / total if total > 0 else float('inf')
            accuracy: float = correct / total if total > 0 else 0.0

            logger.info(
                "Epoch [%d/%d] - Loss: %.4f - Accuracy: %.2f%% - LR: %.6f",
                epoch, self.epochs, avg_loss, accuracy * 100, self.optimizer.param_groups[0]['lr']
            )

        logger.info("Training completed. Final Accuracy: %.2f%%", (correct / total) * 100 if total > 0 else 0.0)
        return self.model


# Sample main function for standalone training (for demonstration purposes).
# In the full project, main.py will handle instantiation and orchestration.
if __name__ == "__main__":
    import sys
    from dataset_loader import DatasetLoader
    from torchvision import models
    from utils import load_config

    # Load configuration.
    config_path: str = "config.yaml"
    config: Dict[str, Any] = load_config(config_path)
    training_config: Dict[str, Any] = config.get("training", {})

    # For demonstration, assume "CIFAR-10" purified data; in practice, use the purified dataset.
    dataset_loader = DatasetLoader(config)
    data_dict = dataset_loader.load_data()
    purified_dataset = data_dict.get("CIFAR-10", {}).get("train", None)
    if purified_dataset is None:
        logger.error("Purified training dataset for CIFAR-10 not found.")
        sys.exit(1)

    # Build ResNet-18 classifier.
    # Modify the final fully connected layer to match the number of classes (assumed to be 10 for CIFAR-10).
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)  # Assuming 10 classes for CIFAR-10

    # Add dataset name in the hyperparams for proper batch size and LR schedule.
    training_config["dataset_name"] = "CIFAR-10"

    # Initialize Trainer and start training.
    trainer = Trainer(model, purified_dataset, training_config)
    trained_model = trainer.train()

    # Optionally, save the trained model.
    torch.save(trained_model.state_dict(), "trained_classifier.pth")
    logger.info("Trained model saved as 'trained_classifier.pth'.")
