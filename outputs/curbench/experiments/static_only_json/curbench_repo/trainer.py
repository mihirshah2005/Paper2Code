"""
trainer.py

This module implements the Trainer class which encapsulates the end-to-end training loop.
It integrates the model (from ModelManager), preprocessed datasets (from DatasetLoader), and a curriculum
learning scheduler (from curriculum.py) using configuration parameters loaded from config.yaml.
The training loop applies curriculum-based loss adjustment on a per-batch basis, logs progress including
average losses per epoch, epoch durations, and GPU memory usage, and returns a summary dictionary with
the total training time and maximum GPU memory consumed.
"""

import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Initialize a logger instance for this module.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Trainer:
    """
    Trainer class encapsulates the training loop for a model using a curriculum learning scheduler.
    
    Attributes:
        model (torch.nn.Module): The neural network model to train.
        data (dict): Dictionary containing at least the "train" DataLoader (and optionally "val", "test", and "num_classes").
        curriculum: An instance of a CurriculumScheduler (subclass) to adjust the training process.
        config (dict): Configuration parameters loaded from config.yaml.
        device (torch.device): Computation device (GPU if available, otherwise CPU).
        optimizer (torch.optim.Optimizer): Optimizer for the training process.
        scheduler (optional): Learning rate scheduler (for example, using cosine annealing) if applicable.
        criterion (torch.nn.Module): Loss function (CrossEntropyLoss).
        domain (str): Inferred training domain ("cv", "nlp", or "graph") based on the model's type.
        num_epochs (int): Number of training epochs as specified in the configuration.
        batch_size (int): Batch size specified for the chosen domain.
    """
    
    def __init__(self, model: nn.Module, data: dict, curriculum, config: dict) -> None:
        """
        Initializes the Trainer instance.
        
        Args:
            model (torch.nn.Module): An instantiated backbone model.
            data (dict): Dictionary with keys "train", "val", "test", and "num_classes" (if applicable).
            curriculum: An instance of a CurriculumScheduler subclass.
            config (dict): Configuration dictionary loaded from config.yaml.
        """
        self.model = model
        self.data = data
        self.curriculum = curriculum
        self.config = config
        
        # Set device (GPU if available, else CPU) and move model to device.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Infer training domain from the model's class name.
        model_class_name = self.model.__class__.__name__.lower()
        if "lenet" in model_class_name or "resnet" in model_class_name or "vit" in model_class_name:
            self.domain = "cv"
        elif "lstm" in model_class_name or "bert" in model_class_name or "gpt2" in model_class_name:
            self.domain = "nlp"
        elif "gcn" in model_class_name or "gat" in model_class_name or "gin" in model_class_name:
            self.domain = "graph"
        else:
            self.domain = "cv"  # Default to CV if not recognized.
        logger.info("Inferred training domain as '%s' based on model type '%s'.", self.domain, self.model.__class__.__name__)
        
        # Retrieve domain-specific training configuration.
        training_config = {}
        if self.domain == "cv":
            training_config = config.get("training", {}).get("cv", {})
        elif self.domain == "nlp":
            # For NLP, choose between 'lstm' and 'transformer' sub-configs.
            if "lstm" in model_class_name:
                training_config = config.get("training", {}).get("nlp", {}).get("lstm", {})
            else:
                training_config = config.get("training", {}).get("nlp", {}).get("transformer", {})
        elif self.domain == "graph":
            training_config = config.get("training", {}).get("graph", {})

        self.num_epochs = int(training_config.get("epochs", 200))
        self.batch_size = int(training_config.get("batch_size", 50))
        
        # Determine optimizer type and learning rate based on the domain.
        optimizer_type = training_config.get("optimizer", "Adam")
        if self.domain == "graph":
            # For graph models, default to TUDataset learning rate if available.
            lr = float(training_config.get("learning_rates", {}).get("TUDataset", 0.01))
        elif self.domain == "nlp" and "lstm" in model_class_name:
            # For LSTM-based NLP models, learning_rate_range is provided as a string like "0.00001-1".
            lr_range = training_config.get("learning_rate_range", "0.00001-1")
            lr = float(lr_range.split("-")[0])
        else:
            lr = float(training_config.get("learning_rate", 0.0001))
        
        # Initialize optimizer based on type.
        if optimizer_type.lower().startswith("adam"):
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer_type.lower().startswith("sgd"):
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
            # For SGD with cosine annealing, initialize the learning rate scheduler.
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.num_epochs)
        elif optimizer_type.lower().startswith("adamw"):
            self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # If no scheduler was set (for optimizers not requiring it), set scheduler to None.
        if not hasattr(self, "scheduler"):
            self.scheduler = None
        
        # Define the loss function to be CrossEntropyLoss for classification.
        self.criterion = nn.CrossEntropyLoss()
        
        # Reset GPU memory counter if using GPU.
        if self.device.type == "cuda":
            torch.cuda.reset_max_memory_allocated(self.device)
        
        logger.info("Trainer initialized with optimizer '%s' (lr=%.6f), num_epochs=%d, batch_size=%d.",
                    optimizer_type, lr, self.num_epochs, self.batch_size)

    def _to_device(self, data_item: any, device: torch.device) -> any:
        """
        Recursively transfers a data item to the specified device.
        
        Args:
            data_item: A tensor, list, tuple, or dict containing tensors.
            device (torch.device): The target device.
        
        Returns:
            The data item with all tensors moved to the target device.
        """
        if isinstance(data_item, torch.Tensor):
            return data_item.to(device)
        elif isinstance(data_item, (list, tuple)):
            return type(data_item)(self._to_device(item, device) for item in data_item)
        elif isinstance(data_item, dict):
            return {key: self._to_device(value, device) for key, value in data_item.items()}
        else:
            return data_item

    def train(self) -> dict:
        """
        Executes the full training loop over the specified number of epochs.
        For each batch, the curriculum scheduler is queried to obtain a scaling factor which is applied to the loss.
        The Trainer logs average loss, epoch duration, and maximum GPU memory usage.
        
        Returns:
            dict: A summary dictionary containing:
                  - "total_time": Total training time in seconds.
                  - "max_gpu_memory_GB": Peak GPU memory consumption in gigabytes (if using GPU; else None).
                  - "final_avg_loss": The average loss of the final epoch.
        """
        self.model.train()
        total_training_start = time.time()
        epoch_losses = []
        
        # Reset GPU memory counter for accurate profiling.
        if self.device.type == "cuda":
            torch.cuda.reset_max_memory_allocated(self.device)
        
        for epoch in range(1, self.num_epochs + 1):
            epoch_start = time.time()
            running_loss = 0.0
            batch_count = 0
            
            for batch in self.data["train"]:
                self.optimizer.zero_grad()
                
                # Process the batch: handle tuple, dict, or graph data objects.
                if isinstance(batch, (list, tuple)):
                    if len(batch) < 2:
                        raise ValueError("Expected batch tuple to contain (inputs, labels).")
                    inputs = self._to_device(batch[0], self.device)
                    labels = self._to_device(batch[1], self.device)
                elif isinstance(batch, dict):
                    # For dictionary batches, expect labels under key 'label' or 'labels'.
                    if "label" in batch:
                        labels = self._to_device(batch["label"], self.device)
                    elif "labels" in batch:
                        labels = self._to_device(batch["labels"], self.device)
                    else:
                        raise ValueError("Label key ('label' or 'labels') not found in batch dictionary.")
                    inputs = self._to_device(batch, self.device)
                else:
                    # Assume batch is a data object (e.g., for graph models) that supports .to(device).
                    batch = batch.to(self.device)
                    inputs = batch
                    # For graph datasets, assume labels are in attribute 'y'.
                    if hasattr(batch, "y"):
                        labels = self._to_device(batch.y, self.device)
                    else:
                        raise ValueError("Graph batch object does not have attribute 'y' for labels.")
                
                # Forward pass.
                outputs = self.model(inputs)
                # For transformer models, check for a 'logits' attribute.
                if hasattr(outputs, "logits"):
                    logits = outputs.logits
                else:
                    logits = outputs
                
                # Compute loss using CrossEntropyLoss.
                loss = self.criterion(logits, labels)
                
                # Curriculum integration: update the scheduler and adjust the loss.
                curriculum_factor = self.curriculum.update_schedule(loss.item(), epoch)
                adjusted_loss = curriculum_factor * loss
                
                # Backpropagation.
                adjusted_loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                batch_count += 1
            
            # Update learning rate scheduler if available.
            if self.scheduler is not None:
                self.scheduler.step()
            
            epoch_duration = time.time() - epoch_start
            avg_loss = running_loss / batch_count if batch_count > 0 else 0.0
            
            max_gpu_memory = None
            if self.device.type == "cuda":
                max_memory_bytes = torch.cuda.max_memory_allocated(self.device)
                max_gpu_memory = max_memory_bytes / (1024 ** 3)  # Convert bytes to GB
            
            logger.info("Epoch %d/%d - Average Loss: %.4f - Duration: %.2f sec - Max GPU Memory: %s GB",
                        epoch, self.num_epochs, avg_loss, epoch_duration,
                        f"{max_gpu_memory:.4f}" if max_gpu_memory is not None else "N/A")
            epoch_losses.append(avg_loss)
        
        total_training_time = time.time() - total_training_start
        final_max_gpu_memory = None
        if self.device.type == "cuda":
            final_max_gpu_memory = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
            
        logger.info("Training complete in %.2f sec.", total_training_time)
        summary = {
            "total_time": total_training_time,
            "max_gpu_memory_GB": final_max_gpu_memory,
            "final_avg_loss": epoch_losses[-1] if epoch_losses else None
        }
        return summary
