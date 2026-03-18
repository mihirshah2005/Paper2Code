"""trainer.py

This module defines the Trainer class that encapsulates the training loop for the 
benchmark experiments. It integrates the model, data loaders (training and validation),
a curriculum scheduler, and experiment configuration to perform training with checkpointing,
loss scaling via curriculum learning, and detailed logging of performance metrics including
epoch loss, duration, and GPU memory usage.

The Trainer class is initialized with:
    - model: a PyTorch model (from ModelManager)
    - data: dictionary containing "train", "val", and "test" datasets
    - curriculum: an instance of a CurriculumScheduler implementing update_schedule(loss, epoch)
    - config: experiment configuration dictionary loaded from config.yaml

The training loop calls the curriculum scheduler on every batch to adjust the loss,
and optionally saves model checkpoints (if configured) in a robust, error-handled manner.
"""

import os
import time
import logging
from typing import Any, Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset

# Set up module-level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)


class Trainer:
    """
    Trainer integrates the model, datasets, curriculum scheduler and configuration
    to execute the training loop with support for checkpointing and logging.

    Attributes:
        model (nn.Module): The backbone model provided by ModelManager.
        data (dict): Dictionary with keys "train", "val", and "test" containing datasets.
        curriculum_scheduler (CurriculumScheduler): Scheduler to adjust training based on loss.
        config (dict): Configuration dictionary loaded from config.yaml.
        train_loader (DataLoader): DataLoader instance for the training set.
        val_loader (DataLoader): DataLoader instance for the validation set.
        device (torch.device): Computation device ("cuda" if available, else "cpu").
        optimizer (torch.optim.Optimizer): Optimizer for model parameter updates.
        criterion (nn.Module): Loss function (typically CrossEntropyLoss).
        epochs (int): Number of training epochs.
        checkpoint_enabled (bool): Flag indicating whether checkpointing is active.
        checkpoint_dir (Optional[str]): Directory for saving checkpoints.
        checkpoint_freq (int): Frequency (in epochs) to save checkpoints.
        training_history (List[Dict[str, Any]]): Per-epoch training statistics.
        domain (str): Task domain ("cv", "nlp", or "graph") determined from the data.
    """

    def __init__(self, 
                 model: nn.Module,
                 data: Dict[str, Any],
                 curriculum_scheduler: Any,
                 config: Dict[str, Any]) -> None:
        """
        Initializes Trainer with provided model, data, curriculum scheduler, and configuration.
        
        Performs the following steps:
            - Detects the task domain from the training dataset.
            - Extracts hyperparameters for the domain from the config.
            - Creates DataLoaders for training and validation sets.
            - Initializes the optimizer and loss function.
            - Moves the model to the appropriate device.
            - Sets up checkpointing based on configuration.
        
        :param model: The neural network model.
        :param data: Dictionary with keys "train", "val", "test" containing datasets.
        :param curriculum_scheduler: An instance implementing update_schedule(loss, epoch).
        :param config: Configuration dictionary.
        """
        self.model = model
        self.data = data
        self.curriculum_scheduler = curriculum_scheduler
        self.config = config
        self.training_history: List[Dict[str, Any]] = []
        
        # Detect domain from training dataset.
        self.domain = self._detect_domain(self.data.get("train"))
        logger.info(f"Detected domain: {self.domain}")

        # Extract training hyperparameters based on domain.
        training_conf = self.config.get("training", {}).get(self.domain, {})
        if self.domain == "cv":
            self.batch_size: int = int(training_conf.get("batch_size", 50))
            self.learning_rate: float = float(training_conf.get("learning_rate", 0.0001))
            self.epochs: int = int(training_conf.get("epochs", 200))
            optimizer_name: str = training_conf.get("optimizer", "Adam")
        elif self.domain == "graph":
            self.batch_size = int(training_conf.get("batch_size", 50))
            self.epochs = int(training_conf.get("epochs", 200))
            # Determine learning rate based on dataset type.
            # Assume if training data has get_idx_split attribute, it is OGB; otherwise, TUDataset.
            if hasattr(self.data.get("train"), "get_idx_split"):
                self.learning_rate = float(training_conf.get("learning_rates", {}).get("OGB", 0.001))
            else:
                self.learning_rate = float(training_conf.get("learning_rates", {}).get("TUDataset", 0.01))
            optimizer_name = training_conf.get("optimizer", "Adam")
        elif self.domain == "nlp":
            # For NLP, decide based on model type.
            model_class_name = type(self.model).__name__
            if model_class_name == "LSTMClassifier":
                lstm_conf = self.config.get("training", {}).get("nlp", {}).get("lstm", {})
                self.batch_size = int(lstm_conf.get("batch_size", 50))
                # For LSTM, use SGD with cosine annealing possibility.
                # Here we use torch.optim.SGD and note that a cosine annealing scheduler may be applied externally.
                lr_range = lstm_conf.get("learning_rate_range", "0.00001-1")
                # We take the lower bound as the learning rate.
                self.learning_rate = float(lr_range.split('-')[0])
                self.epochs = int(lstm_conf.get("epochs", 10))
                optimizer_name = lstm_conf.get("optimizer", "SGD")
            else:
                transformer_conf = self.config.get("training", {}).get("nlp", {}).get("transformer", {})
                self.batch_size = int(transformer_conf.get("batch_size", 50))
                self.learning_rate = float(transformer_conf.get("learning_rate", 0.00002))
                self.epochs = int(transformer_conf.get("epochs", 3))
                optimizer_name = transformer_conf.get("optimizer", "AdamW")
        else:
            logger.error(f"Unsupported domain {self.domain} during Trainer initialization.")
            raise ValueError(f"Unsupported domain {self.domain}")

        logger.info(f"Training hyperparameters: batch_size={self.batch_size}, learning_rate={self.learning_rate}, "
                    f"epochs={self.epochs}, optimizer={optimizer_name}")

        # Create DataLoaders for training and validation sets.
        self.train_loader = DataLoader(self.data.get("train"), batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.val_loader = DataLoader(self.data.get("val"), batch_size=self.batch_size, shuffle=False, num_workers=0)

        # Set device.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logger.info(f"Using device: {self.device}")

        # Initialize loss function (CrossEntropyLoss for classification).
        self.criterion = nn.CrossEntropyLoss()

        # Initialize optimizer based on optimizer_name.
        if "adamw" in optimizer_name.lower():
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        elif "sgd" in optimizer_name.lower():
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        else:
            # Default to Adam.
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Setup checkpointing.
        self.checkpoint_enabled = False
        checkpoint_conf = self.config.get("checkpoint", {})
        self.checkpoint_dir: Optional[str] = checkpoint_conf.get("dir")
        self.checkpoint_freq: int = int(checkpoint_conf.get("frequency", 0))
        if self.checkpoint_dir and self.checkpoint_freq > 0:
            try:
                os.makedirs(self.checkpoint_dir, exist_ok=True)
                # Verify that we can write to this directory.
                test_path = os.path.join(self.checkpoint_dir, "temp.txt")
                with open(test_path, "w") as f:
                    f.write("test")
                os.remove(test_path)
                self.checkpoint_enabled = True
                logger.info(f"Checkpointing enabled. Directory: {self.checkpoint_dir}, Frequency: {self.checkpoint_freq} epochs.")
            except Exception as e:
                logger.warning(f"Checkpointing disabled due to error with directory '{self.checkpoint_dir}': {e}")
                self.checkpoint_enabled = False
        else:
            logger.info("Checkpointing not configured; proceeding without saving checkpoints.")

    def _detect_domain(self, train_data: Any) -> str:
        """
        Detects the task domain based on the training dataset.

        Heuristics:
            - If train_data is a list (or first element is a dict), assume NLP.
            - If train_data (or underlying dataset if Subset) has 'targets' attribute, assume CV.
            - Otherwise, if the dataset (or underlying dataset) has attribute 'num_node_features', assume Graph.
        
        :param train_data: The training dataset.
        :return: Domain string: "cv", "nlp", or "graph".
        """
        if isinstance(train_data, list):
            return "nlp"
        # If train_data is a Subset, try to get the underlying dataset.
        dataset_obj = train_data
        if isinstance(train_data, Subset) and hasattr(train_data, "dataset"):
            dataset_obj = train_data.dataset

        if hasattr(dataset_obj, "targets"):
            return "cv"
        # For graph datasets, typically they are Torch Geometric datasets which have num_node_features.
        if hasattr(dataset_obj, "num_node_features") or (hasattr(dataset_obj, "__getitem__") and hasattr(dataset_obj[0], "x")):
            return "graph"
        # Fallback default to CV.
        logger.warning("Unable to definitively determine domain from training data; defaulting to 'cv'.")
        return "cv"

    def _validate(self) -> float:
        """
        Runs a validation pass on the validation set and computes the average loss.
        
        :return: Average validation loss for the epoch.
        """
        self.model.eval()
        val_loss_total: float = 0.0
        batch_count: int = 0

        with torch.no_grad():
            for batch in self.val_loader:
                inputs, labels = self._unpack_batch(batch)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                val_loss_total += loss.item()
                batch_count += 1

        avg_val_loss = val_loss_total / batch_count if batch_count > 0 else 0.0
        return avg_val_loss

    def _unpack_batch(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Unpacks a batch from the DataLoader into input tensor and label tensor.
        Checks if batch is a tuple/list; otherwise, attempts dictionary unpacking.
        
        :param batch: Batch data from DataLoader.
        :return: Tuple of (inputs, labels) as Tensors.
        """
        if isinstance(batch, (list, tuple)):
            inputs, labels = batch
        elif isinstance(batch, dict):
            # For dict batch, assume keys "input_ids" and "labels" or "data" and "target"
            if "input_ids" in batch and "labels" in batch:
                inputs, labels = batch["input_ids"], batch["labels"]
            elif "data" in batch and "target" in batch:
                inputs, labels = batch["data"], batch["target"]
            else:
                logger.error("Unrecognized batch dictionary format.")
                raise ValueError("Batch dictionary does not contain recognized keys.")
        else:
            logger.error("Unrecognized batch format; expected tuple, list, or dict.")
            raise ValueError("Unable to unpack batch data.")

        # Transfer to device.
        inputs = self._to_device(inputs)
        labels = self._to_device(labels)
        return inputs, labels

    def _to_device(self, data: Any) -> Any:
        """
        Recursively moves data to the configured device. Supports tensors, lists, and dictionaries.
        
        :param data: Data to be moved.
        :return: Data on the trainer's device.
        """
        if torch.is_tensor(data):
            return data.to(self.device)
        elif isinstance(data, list):
            return [self._to_device(item) for item in data]
        elif isinstance(data, dict):
            return {key: self._to_device(val) for key, val in data.items()}
        else:
            return data

    def train(self) -> Tuple[nn.Module, List[Dict[str, Any]]]:
        """
        Executes the training pipeline over the specified number of epochs.

        Training process per epoch:
            - Records epoch start time.
            - Resets GPU memory counter if applicable.
            - Iterates over training batches:
                * Unpacks batch data.
                * Performs forward pass and computes loss.
                * Invokes curriculum_scheduler.update_schedule(loss, epoch) to get a weighting factor.
                * Adjusts the loss if a valid float factor is returned.
                * Backpropagates and updates model parameters.
                * Logs per-batch metrics.
            - Computes average training loss, epoch duration, and maximum GPU memory usage.
            - Optionally runs a validation pass.
            - Logs an epoch summary.
            - Saves checkpoints as configured.
        
        After training, logs total training time and returns the final model and training metrics.

        :return: Tuple containing the trained model and a list of per-epoch training metrics.
        """
        overall_start_time = time.time()
        device_name = self.device.type

        if torch.cuda.is_available():
            torch.cuda.reset_max_memory_allocated(self.device)

        for epoch in range(self.epochs):
            epoch_start = time.time()
            self.model.train()
            running_loss: float = 0.0
            batch_count: int = 0
            curriculum_factors: List[float] = []

            for batch in self.train_loader:
                try:
                    inputs, labels = self._unpack_batch(batch)
                except ValueError as e:
                    logger.error(f"Error unpacking batch: {e}")
                    continue

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # Obtain curriculum factor from scheduler.
                factor = self.curriculum_scheduler.update_schedule(loss.item(), epoch)
                if isinstance(factor, float):
                    adjusted_loss = loss * factor
                    curriculum_factors.append(factor)
                elif factor is None:
                    adjusted_loss = loss
                else:
                    logger.warning(f"Unexpected return type from update_schedule: {type(factor)}. Using original loss.")
                    adjusted_loss = loss

                adjusted_loss.backward()
                self.optimizer.step()

                running_loss += adjusted_loss.item()
                batch_count += 1

                logger.debug(f"Epoch {epoch+1}, Batch {batch_count}: Loss = {loss.item():.4f}, "
                             f"Curriculum factor = {factor if isinstance(factor, float) else 'N/A'}.")

            avg_train_loss = running_loss / batch_count if batch_count > 0 else 0.0
            epoch_duration = time.time() - epoch_start

            # Measure peak GPU memory usage in GB (if using CUDA).
            if torch.cuda.is_available():
                max_memory_bytes = torch.cuda.max_memory_allocated(self.device)
                max_memory_gb = max_memory_bytes / (1024 ** 3)
            else:
                max_memory_gb = 0.0

            # Run validation pass.
            avg_val_loss = self._validate()

            # Log epoch summary.
            epoch_summary = {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "duration_sec": epoch_duration,
                "max_gpu_memory_gb": max_memory_gb,
                "avg_curriculum_factor": (sum(curriculum_factors) / len(curriculum_factors)) if curriculum_factors else None
            }
            self.training_history.append(epoch_summary)
            logger.info(f"Epoch [{epoch+1}/{self.epochs}] - Train Loss: {avg_train_loss:.4f}, "
                        f"Val Loss: {avg_val_loss:.4f}, Duration: {epoch_duration:.2f}s, "
                        f"Max GPU Memory: {max_memory_gb:.4f} GB, "
                        f"Avg Curriculum Factor: {epoch_summary['avg_curriculum_factor']}.")

            # Reset GPU memory counter for next epoch if using CUDA.
            if torch.cuda.is_available():
                torch.cuda.reset_max_memory_allocated(self.device)

            # Checkpointing: Save checkpoint if enabled and if the frequency condition is met.
            if self.checkpoint_enabled and ((epoch + 1) % self.checkpoint_freq == 0 or (epoch + 1) == self.epochs):
                checkpoint_path = os.path.join(self.checkpoint_dir, f"model_epoch_{epoch+1}.pth")
                try:
                    torch.save({
                        "epoch": epoch + 1,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "training_history": self.training_history,
                    }, checkpoint_path)
                    logger.info(f"Checkpoint saved at epoch {epoch+1} to {checkpoint_path}.")
                except Exception as e:
                    logger.warning(f"Failed to save checkpoint at epoch {epoch+1}: {e}")

        total_training_time = time.time() - overall_start_time
        logger.info(f"Training complete. Total training time: {total_training_time:.2f} seconds on device {device_name}.")

        # Optionally, save final model checkpoint.
        if self.checkpoint_enabled:
            final_checkpoint_path = os.path.join(self.checkpoint_dir, "model_final.pth")
            try:
                torch.save({
                    "epoch": self.epochs,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "training_history": self.training_history,
                }, final_checkpoint_path)
                logger.info(f"Final model checkpoint saved to {final_checkpoint_path}.")
            except Exception as e:
                logger.warning(f"Failed to save final model checkpoint: {e}")

        return self.model, self.training_history
