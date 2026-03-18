"""trainer.py

This module defines the Trainer class responsible for training a classifier (ResNet-18)
on a purified training dataset. The trainer uses hyperparameters from a configuration file 
(config.yaml) and implements training using the SGD optimizer with a MultiStepLR scheduler.
It logs the training loss and accuracy per epoch and returns the final trained model along 
with training history.

Author: [Your Name]
Date: [Date]
"""

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import load_config, get_logger  # Ensure utils.py is available


class Trainer:
    """
    Trainer class to train a classifier on a purified dataset.
    
    Attributes:
        model (torch.nn.Module): The classifier model to be trained.
        train_dataset (torch.utils.data.Dataset): The purified training dataset.
        hyperparams (Dict[str, Any]): Dictionary of hyperparameters from config.yaml.
        dataset_name (str): Name of the dataset ("CIFAR-10", "ImageNet-100", or "ImageNet-Dog").
        epochs (int): Total training epochs.
        initial_lr (float): Initial learning rate.
        batch_size (int): Batch size for training.
        lr_milestones (List[int]): List of epoch milestones at which to decay the LR.
        optimizer_type (str): Optimizer type (expected "SGD").
        device (torch.device): Device to run training on.
        train_loader (DataLoader): DataLoader wrapping the purified training dataset.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        criterion (nn.Module): Loss function (CrossEntropyLoss).
        scheduler (torch.optim.lr_scheduler): LR scheduler with milestones.
        logger (logging.Logger): Logger instance.
    """

    def __init__(self, model: torch.nn.Module, train_dataset: torch.utils.data.Dataset,
                 hyperparams: Dict[str, Any], dataset_name: str) -> None:
        """
        Initializes the Trainer with the specified model, training dataset, hyperparameters,
        and dataset identifier.

        Args:
            model (torch.nn.Module): The classifier model to be trained.
            train_dataset (torch.utils.data.Dataset): Purified training dataset.
            hyperparams (Dict[str, Any]): Hyperparameters dictionary from configuration.
            dataset_name (str): Name of the dataset ("CIFAR-10", "ImageNet-100", or "ImageNet-Dog").
        """
        self.logger = get_logger("Trainer")
        self.model = model
        self.dataset_name = dataset_name

        # Extract training hyperparameters from the configuration.
        training_config: Dict[str, Any] = hyperparams.get("training", {})
        self.epochs: int = int(training_config.get("epochs", 100))
        self.initial_lr: float = float(training_config.get("learning_rate", 0.1))
        self.optimizer_type: str = training_config.get("optimizer", "SGD")

        # Batch size specific to the dataset.
        batch_size_config: Dict[str, Any] = training_config.get("batch_size", {})
        self.batch_size: int = int(batch_size_config.get(dataset_name, 128))

        # Learning rate decay schedule: list of milestone epochs.
        lr_decay_schedule: Dict[str, Any] = training_config.get("lr_decay_schedule", {})
        # Default milestones if not available.
        default_milestones: List[int] = [50, 75] if dataset_name == "CIFAR-10" else [30, 60]
        # Attempt to extract milestones from config; each item is expected to be a dict with key "epoch".
        milestones_raw = lr_decay_schedule.get(dataset_name, default_milestones)
        if isinstance(milestones_raw, list) and all(isinstance(item, dict) for item in milestones_raw):
            self.lr_milestones: List[int] = [int(item.get("epoch", 0)) for item in milestones_raw]
        else:
            # If not in expected dict-list format, try to convert directly.
            self.lr_milestones = [int(x) for x in milestones_raw] if isinstance(milestones_raw, list) else default_milestones

        self.logger.info(f"Dataset: {dataset_name} | Epochs: {self.epochs} | Batch Size: {self.batch_size} | "
                         f"Initial LR: {self.initial_lr} | Milestones: {self.lr_milestones}")

        # Setup device.
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Create DataLoader for the training dataset.
        self.train_loader: DataLoader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,  # You may adjust this based on your system.
            pin_memory=True if torch.cuda.is_available() else False
        )

        # Setup optimizer.
        if self.optimizer_type.upper() == "SGD":
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.initial_lr, momentum=0.9)
        else:
            # Default to SGD if an unknown optimizer type is specified.
            self.logger.warning(f"Optimizer type '{self.optimizer_type}' not recognized. Defaulting to SGD.")
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.initial_lr, momentum=0.9)
        
        # Loss function: Cross Entropy Loss.
        self.criterion: nn.Module = nn.CrossEntropyLoss()

        # Setup learning rate scheduler with MultiStepLR.
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Initialize training history.
        self.history: Dict[str, List[float]] = {
            "loss": [],
            "accuracy": [],
            "lr": []
        }

    def train(self) -> Tuple[torch.nn.Module, Dict[str, List[float]]]:
        """
        Trains the model over the specified number of epochs using the purified dataset.
        Logs the epoch loss, accuracy, and current learning rate.

        Returns:
            Tuple[torch.nn.Module, Dict[str, List[float]]]: The final trained model and the training history.
        """
        self.logger.info("Starting training...")
        total_samples: int = len(self.train_loader.dataset)

        for epoch in range(self.epochs):
            self.model.train()
            running_loss: float = 0.0
            correct_predictions: int = 0
            samples_count: int = 0

            for batch_idx, batch in enumerate(self.train_loader):
                # Expect batch to be a tuple (inputs, labels) or a dict containing "image" and "label"
                if isinstance(batch, dict):
                    inputs = batch.get("image")
                    labels = batch.get("label")
                elif isinstance(batch, (list, tuple)):
                    # Assume first element as inputs and second as labels.
                    inputs, labels = batch[0], batch[1]
                else:
                    self.logger.error("Unsupported batch format in training data.")
                    continue
                
                # Move data to device.
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Zero the gradients.
                self.optimizer.zero_grad()

                # Forward pass.
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # Backward pass and optimization step.
                loss.backward()
                self.optimizer.step()

                # Update running loss.
                batch_size_current: int = inputs.size(0)
                running_loss += loss.item() * batch_size_current
                samples_count += batch_size_current

                # Compute correct predictions in this batch.
                predictions = outputs.argmax(dim=1)
                correct_predictions += (predictions == labels).sum().item()

            epoch_loss: float = running_loss / samples_count
            epoch_accuracy: float = correct_predictions / samples_count
            current_lr: float = self.optimizer.param_groups[0]["lr"]

            # Log metrics for the epoch.
            self.logger.info(
                f"Epoch {epoch + 1}/{self.epochs}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_accuracy:.4f}, LR = {current_lr:.6f}"
            )

            # Append metrics to history.
            self.history["loss"].append(epoch_loss)
            self.history["accuracy"].append(epoch_accuracy)
            self.history["lr"].append(current_lr)

            # Step the learning rate scheduler.
            self.scheduler.step()

        self.logger.info("Training completed.")
        return self.model, self.history


# For standalone testing of the Trainer module, one could insert:
if __name__ == "__main__":
    # Example usage (for debugging purposes):
    # Load configuration.
    config = load_config("config.yaml")
    # For demonstration, assume a purified dataset is available as a PyTorch Dataset.
    # Here we create a dummy dataset using random tensors.
    from torch.utils.data import TensorDataset

    # Set dataset_name to "CIFAR-10" for this example.
    dataset_name = "CIFAR-10"
    batch_size = int(config.get("training", {}).get("batch_size", {}).get(dataset_name, 128))
    
    dummy_inputs = torch.randn(1000, 3, 32, 32)  # 1000 samples of CIFAR-10 like images.
    dummy_labels = torch.randint(low=0, high=10, size=(1000,))  # 10 classes.
    purified_dataset = TensorDataset(dummy_inputs, dummy_labels)

    # Create a dummy ResNet-18. For simplicity, use torchvision's resnet18.
    import torchvision.models as models
    model = models.resnet18(num_classes=10)

    # Instantiate Trainer.
    trainer = Trainer(model=model, train_dataset=purified_dataset, hyperparams=config, dataset_name=dataset_name)
    # Run training.
    trained_model, history = trainer.train()

    # Print training history.
    print("Training History:", history)
