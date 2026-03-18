"""trainer.py

This module implements the Trainer class, which encapsulates the training loop
for curriculum learning across different domains (CV, NLP, Graph). The Trainer
integrates curriculum-based adjustments into the loss computation and tracks
performance and resource usage during training.

It expects the following inputs:
  - model: The PyTorch model (as built by ModelManager).
  - data: A dictionary containing DataLoader objects for "train", "val", and "test".
  - curriculum: An instance of a CurriculumScheduler (e.g. SelfPacedScheduler,
      TeacherGuidedScheduler, LossReweightingScheduler).
  - config: The experiment configuration dictionary loaded from config.yaml.

The training loop handles different types of curriculum adjustment outputs:
  • A scalar (int/float or one-element tensor) is interpreted as a uniform weight.
  • A tensor of shape (batch_size,) is used for per-sample loss weighting.
  • A list (of booleans or integer indices) is used to select a subset of samples.
If the adjustment type is not recognized, no adjustment is applied.
"""

import time
import logging
from typing import Any, Dict, List, Union

import torch
import torch.nn as nn
import torch.optim as optim

from curriculum import CurriculumScheduler  # local import

# Configure module-level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Trainer:
    """Encapsulates the training loop and integrates curriculum learning strategies.

    Attributes:
        model (nn.Module): The model to be trained.
        data (Dict[str, Any]): Dict containing DataLoader objects under keys "train", "val", and "test".
        curriculum (CurriculumScheduler): The curriculum learning scheduler instance.
        config (Dict[str, Any]): The configuration dictionary loaded from config.yaml.
        optimizer (torch.optim.Optimizer): The optimizer used for updating model parameters.
        criterion (nn.Module): The loss function (CrossEntropyLoss with reduction 'none').
        device (torch.device): The computation device (GPU or CPU).
        domain (str): Inferred domain ("cv", "nlp", or "graph").
        epochs (int): Number of training epochs.
    """

    def __init__(self, model: nn.Module, data: Dict[str, Any],
                 curriculum: CurriculumScheduler, config: Dict[str, Any]) -> None:
        """Initializes the Trainer with the model, data, curriculum scheduler, and config.

        Args:
            model (nn.Module): The backbone model to be trained.
            data (Dict[str, Any]): Dictionary of DataLoader objects for "train", "val", and "test".
            curriculum (CurriculumScheduler): An instance of a curriculum scheduler.
            config (Dict[str, Any]): Experiment configuration dictionary.
        """
        self.model = model
        self.data = data
        self.curriculum = curriculum
        self.config = config

        # Set computation device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Infer domain from the dataset identifier in config; default is "cifar10"
        dataset_name = str(self.config.get("dataset", "cifar10")).lower()
        if dataset_name in {"cifar10", "cifar100", "tinyimagenet"}:
            self.domain = "cv"
        elif dataset_name in {"rte", "mrpc", "stsb", "cola", "sst2", "qnli", "qqp", "mnli"}:
            self.domain = "nlp"
        elif dataset_name in {"ogbg-molhiv", "mutag", "proteins", "nci1"}:
            self.domain = "graph"
        else:
            self.domain = "cv"  # default

        # Set training hyperparameters based on the domain from config
        if self.domain == "cv":
            training_config = self.config.get("training", {}).get("cv", {})
            self.epochs = int(training_config.get("epochs", 200))
            learning_rate = float(training_config.get("learning_rate", 0.0001))
            optimizer_type = training_config.get("optimizer", "Adam")
        elif self.domain == "nlp":
            # Decide based on the model class name; if it's LSTM then use 'lstm' family otherwise transformer.
            model_name_lower = self.model.__class__.__name__.lower()
            nlp_config = self.config.get("training", {}).get("nlp", {})
            if "lstmclassifier" in model_name_lower:
                lstm_config = nlp_config.get("lstm", {})
                self.epochs = int(lstm_config.get("epochs", 10))
                # learning_rate_range is provided as "0.00001-1"; take the lower bound as LR.
                lr_range = lstm_config.get("learning_rate_range", "0.00001-1")
                lr_lower = float(lr_range.split("-")[0])
                learning_rate = lr_lower
                optimizer_type = lstm_config.get("optimizer", "SGD")
            else:
                transformer_config = nlp_config.get("transformer", {})
                self.epochs = int(transformer_config.get("epochs", 3))
                learning_rate = float(transformer_config.get("learning_rate", 0.00002))
                optimizer_type = transformer_config.get("optimizer", "AdamW")
        elif self.domain == "graph":
            training_config = self.config.get("training", {}).get("graph", {})
            self.epochs = int(training_config.get("epochs", 200))
            if dataset_name == "ogbg-molhiv":
                # Use OGB specific learning rate from config
                learning_rate = float(training_config.get("learning_rates", {}).get("OGB", 0.001))
            else:
                learning_rate = float(training_config.get("learning_rates", {}).get("TUDataset", 0.01))
            optimizer_type = training_config.get("optimizer", "Adam")
        else:
            self.epochs = 200
            learning_rate = 0.0001
            optimizer_type = "Adam"

        # Initialize optimizer based on the provided type
        if optimizer_type.lower() == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_type.lower() == "adamw":
            self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        elif "sgd" in optimizer_type.lower():
            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        else:
            logger.warning(f"Unsupported optimizer '{optimizer_type}', defaulting to Adam.")
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Initialize loss function with reduction set to 'none' for per-sample loss computations.
        self.criterion = nn.CrossEntropyLoss(reduction="none")

        logger.info(
            f"Trainer initialized: Domain: {self.domain}, Epochs: {self.epochs}, "
            f"Learning Rate: {learning_rate}, Optimizer: {optimizer_type}, Device: {self.device}"
        )

    def train(self) -> nn.Module:
        """Executes the training loop with curriculum integration.

        For each epoch, the Trainer iterates through the training batches, computes
        per-sample losses, applies curriculum adjustments (via scalar weight, per-sample
        weights, or selection indices), backpropagates the adjusted loss, and updates
        the model parameters. Epoch-level metrics (loss, duration, GPU memory usage) are logged.

        Returns:
            nn.Module: The trained model.
        """
        self.model.train()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        global_start_time = time.time()

        for epoch in range(1, self.epochs + 1):
            epoch_start_time = time.time()
            epoch_loss_total = 0.0
            batch_count = 0

            for batch in self.data.get("train", []):
                # Data preparation: support for tuple/list or dict batches.
                if isinstance(batch, (list, tuple)):
                    xb, yb = batch[0], batch[1]
                    if isinstance(xb, torch.Tensor):
                        xb = xb.to(self.device)
                    if isinstance(yb, torch.Tensor):
                        yb = yb.to(self.device)
                elif isinstance(batch, dict):
                    xb = {k: v.to(self.device) for k, v in batch.items() if k != "labels"}
                    yb = batch.get("labels")
                    if isinstance(yb, torch.Tensor):
                        yb = yb.to(self.device)
                else:
                    logger.error("Unrecognized batch format. Skipping batch.")
                    continue

                # Zero gradients
                self.optimizer.zero_grad()

                # Forward pass via model(x)
                outputs = self.model(xb)

                # Compute base loss with reduction 'none' for per-sample value
                base_loss = self.criterion(outputs, yb)
                current_loss_value = base_loss.mean().item()

                # Obtain curriculum adjustment information
                curriculum_adjustment = self.curriculum.update_schedule(
                    current_loss_value, epoch, inputs=xb, labels=yb
                )
                effective_loss: Union[torch.Tensor, float, None] = None

                # Branch on the type of curriculum_adjustment
                if isinstance(curriculum_adjustment, (int, float)):
                    # Uniform scalar weight
                    scalar_weight = float(curriculum_adjustment)
                    effective_loss = scalar_weight * base_loss.mean()
                elif isinstance(curriculum_adjustment, torch.Tensor):
                    if curriculum_adjustment.numel() == 1:
                        scalar_weight = curriculum_adjustment.item()
                        effective_loss = scalar_weight * base_loss.mean()
                    elif curriculum_adjustment.shape[0] == base_loss.shape[0]:
                        effective_loss = (curriculum_adjustment * base_loss).mean()
                    else:
                        logger.warning(
                            "Curriculum adjustment tensor shape does not match batch size; defaulting to no adjustment."
                        )
                        effective_loss = base_loss.mean()
                elif isinstance(curriculum_adjustment, list):
                    # Interpret as a boolean mask or list of indices
                    try:
                        if all(isinstance(item, bool) for item in curriculum_adjustment):
                            mask = torch.tensor(curriculum_adjustment, dtype=torch.bool, device=self.device)
                            if mask.sum() == 0:
                                logger.warning("Curriculum adjustment resulted in no selected samples; skipping batch.")
                                continue
                            effective_loss = base_loss[mask].mean()
                        elif all(isinstance(item, int) for item in curriculum_adjustment):
                            indices = torch.tensor(curriculum_adjustment, device=self.device)
                            effective_loss = base_loss[indices].mean()
                        else:
                            logger.error("Unrecognized list type in curriculum adjustment; defaulting to no adjustment.")
                            effective_loss = base_loss.mean()
                    except Exception as e:
                        logger.error(f"Error processing curriculum adjustment list: {e}; defaulting to no adjustment.")
                        effective_loss = base_loss.mean()
                else:
                    logger.error("Unrecognized type from curriculum scheduler; defaulting to no adjustment.")
                    effective_loss = base_loss.mean()

                # Backward pass and parameter update
                effective_loss.backward()
                self.optimizer.step()

                epoch_loss_total += effective_loss.item()
                batch_count += 1

            # Epoch metrics and GPU memory usage
            epoch_duration = time.time() - epoch_start_time
            avg_epoch_loss = epoch_loss_total / (batch_count if batch_count > 0 else 1)
            gpu_memory_gb = 0.0
            if self.device.type == "cuda":
                gpu_memory_bytes = torch.cuda.max_memory_allocated(self.device)
                gpu_memory_gb = gpu_memory_bytes / (1024 ** 3)
                torch.cuda.reset_peak_memory_stats(self.device)

            logger.info(
                f"Epoch [{epoch}/{self.epochs}]: Avg Loss: {avg_epoch_loss:.4f}, "
                f"Duration: {epoch_duration:.2f}s, GPU Memory: {gpu_memory_gb:.2f}GB"
            )

        total_duration = time.time() - global_start_time
        logger.info(f"Training completed in {total_duration:.2f}s over {self.epochs} epochs.")
        return self.model
