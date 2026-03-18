"""trainer.py

This module defines the Trainer class which manages the training loop for the
Transformer model as described in "Attention Is All You Need." It handles optimizer
and learning rate scheduling (with warmup), loss computation with label smoothing,
gradient clipping, periodic validation, and checkpoint saving. All hyperparameters
and settings are read from a configuration dictionary (parsed from config.yaml).

Author: [Your Name]
Date: [Current Date]
"""

import os
import time
import math
import logging
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

# Setup module-level logger in a reproducible manner.
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
    """Trainer class to manage training loop including optimizer, learning rate scheduling,
    gradient clipping, validation, and checkpoint saving.

    Attributes:
        model (nn.Module): The Transformer model.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        config (Dict[str, Any]): Configuration dictionary parsed from config.yaml.
        device (torch.device): Device to run the model on.
        global_step (int): Counter for training steps.
        total_steps (int): Total number of training steps.
        warmup_steps (int): Number of warmup steps for the learning rate scheduler.
        d_model (int): Model dimension.
        label_smoothing (float): Label smoothing parameter.
        gradient_clip (float): Maximum norm for gradient clipping.
        optimizer (torch.optim.Optimizer): Optimizer instance.
        checkpoint_interval (float): Checkpoint interval in seconds.
        avg_checkpoints (int): Number of recent checkpoints to average (for inference later).
        last_checkpoint_time (float): Timestamp when the last checkpoint was saved.
    """

    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                 config: Dict[str, Any]) -> None:
        """
        Initializes Trainer with the provided model, training and validation data loaders,
        and configuration dictionary.
        """
        # Save configuration.
        self.config: Dict[str, Any] = config

        # Set device from torch.cuda or CPU.
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: nn.Module = model.to(self.device)
        self.train_loader: DataLoader = train_loader
        self.val_loader: DataLoader = val_loader

        # Determine model type ("base" or "big"). Default is "base".
        self.model_type: str = config.get("model_type", "base").lower()

        # Validate and extract training hyperparameters.
        training_cfg: Dict[str, Any] = config.get("training", {})
        self.beta1: float = training_cfg.get("beta1", 0.9)
        self.beta2: float = training_cfg.get("beta2", 0.98)
        self.epsilon: float = training_cfg.get("epsilon", 1e-9)
        self.warmup_steps: int = training_cfg.get("warmup_steps", 4000)
        self.label_smoothing: float = training_cfg.get("label_smoothing", 0.1)
        # Define a default gradient clipping parameter.
        self.gradient_clip: float = training_cfg.get("gradient_clip", 1.0)

        # Set total training steps depending on model type.
        total_steps_cfg = training_cfg.get("total_steps", {})
        if self.model_type == "big":
            self.total_steps: int = total_steps_cfg.get("big_model", 300000)
        else:
            self.total_steps: int = total_steps_cfg.get("base_model", 100000)

        # Extract model-related hyperparameters.
        model_cfg: Dict[str, Any] = config.get("model", {}).get(self.model_type, {})
        # d_model is critical for learning rate scheduling.
        self.d_model: int = model_cfg.get("d_model", 512)

        # Initialize the optimizer with dummy initial learning rate (will be updated in each step).
        self.optimizer: torch.optim.Optimizer = Adam(
            self.model.parameters(),
            lr=0.0,
            betas=(self.beta1, self.beta2),
            eps=self.epsilon
        )

        # Setup checkpoint parameters.
        checkpoint_cfg: Dict[str, Any] = config.get("checkpoint", {})
        # checkpoint_interval is given in minutes; convert to seconds.
        self.checkpoint_interval: float = checkpoint_cfg.get("checkpoint_interval_minutes", 10) * 60.0
        avg_ckpt_cfg = checkpoint_cfg.get("average_checkpoints", {})
        if self.model_type == "big":
            self.avg_checkpoints: int = avg_ckpt_cfg.get("big", 20)
        else:
            self.avg_checkpoints: int = avg_ckpt_cfg.get("base", 5)

        self.global_step: int = 0
        self.last_checkpoint_time: float = time.time()

        # Log the configuration details.
        logger.info(f"Trainer initialized on device {self.device}. "
                    f"Model type: {self.model_type}, d_model: {self.d_model}, total_steps: {self.total_steps}, "
                    f"warmup_steps: {self.warmup_steps}, label_smoothing: {self.label_smoothing}, "
                    f"gradient_clip: {self.gradient_clip}")

    def _get_learning_rate(self, step: int) -> float:
        """
        Compute the learning rate using the schedule:
            lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5)).
        Args:
            step (int): Current training step (should be > 0).
        Returns:
            float: The computed learning rate.
        """
        effective_step: float = float(step if step > 0 else 1)
        lr: float = (self.d_model ** (-0.5)) * min(effective_step ** (-0.5),
                                                    effective_step * (self.warmup_steps ** (-1.5)))
        return lr

    def save_checkpoint(self) -> None:
        """
        Saves a checkpoint containing the model state_dict, optimizer state_dict, global step,
        and configuration. The checkpoint is saved to the "checkpoints" directory.
        """
        checkpoint_dir: str = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path: str = os.path.join(checkpoint_dir, f"checkpoint_step_{self.global_step}.pt")
        checkpoint: Dict[str, Any] = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "config": self.config
        }
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint at step {self.global_step} to {checkpoint_path}")

    def validate(self) -> float:
        """
        Runs validation over the validation DataLoader and computes the average loss.
        Returns:
            float: Average validation loss.
        """
        self.model.eval()
        total_loss: float = 0.0
        count: int = 0
        pad_index: int = 0  # Assuming pad token index is 0.
        with torch.no_grad():
            for batch in self.val_loader:
                if "src" in batch:
                    src = batch["src"].to(self.device)
                    tgt = batch["tgt"].to(self.device)
                    src_mask = batch["src_mask"].to(self.device)
                    tgt_mask = batch["tgt_mask"].to(self.device)
                elif "input" in batch:
                    # For parsing tasks, use the same input as source and target.
                    src = batch["input"].to(self.device)
                    tgt = batch["input"].to(self.device)
                    src_mask = batch["input_mask"].to(self.device)
                    tgt_mask = batch["input_mask"].to(self.device)
                else:
                    raise ValueError("Batch does not contain expected keys.")

                logits = self.model(src, tgt, src_mask, tgt_mask)
                logits_flat = logits.view(-1, logits.size(-1))
                tgt_flat = tgt.view(-1)
                loss = F.cross_entropy(
                    logits_flat,
                    tgt_flat,
                    ignore_index=pad_index,
                    label_smoothing=self.label_smoothing
                )
                total_loss += loss.item()
                count += 1
        self.model.train()
        avg_loss: float = total_loss / count if count > 0 else float('inf')
        return avg_loss

    def train(self) -> None:
        """
        Executes the training loop until the total number of specified steps is reached.
        For each batch, the function performs a forward pass, computes loss with label smoothing,
        applies gradient clipping, updates the learning rate, and steps the optimizer. It logs
        training metrics periodically, runs validation intermittently, and saves checkpoints based
        on the configured time interval.
        """
        self.model.train()
        start_time: float = time.time()
        # Create an iterator for the train_loader.
        train_iter = iter(self.train_loader)
        while self.global_step < self.total_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                # Restart loader if epoch finishes.
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            # Move batch to device. Determine task type by key presence.
            if "src" in batch:
                src = batch["src"].to(self.device)
                tgt = batch["tgt"].to(self.device)
                src_mask = batch["src_mask"].to(self.device)
                tgt_mask = batch["tgt_mask"].to(self.device)
            elif "input" in batch:
                # For parsing tasks, treat the input as both source and target.
                src = batch["input"].to(self.device)
                tgt = batch["input"].to(self.device)
                src_mask = batch["input_mask"].to(self.device)
                tgt_mask = batch["input_mask"].to(self.device)
            else:
                raise ValueError("Batch does not contain expected keys.")

            # Forward pass.
            logits = self.model(src, tgt, src_mask, tgt_mask)
            # Reshape logits and target for loss computation.
            pad_index: int = 0  # Assuming pad token index is 0.
            logits_flat = logits.view(-1, logits.size(-1))
            tgt_flat = tgt.view(-1)
            loss = F.cross_entropy(
                logits_flat,
                tgt_flat,
                ignore_index=pad_index,
                label_smoothing=self.label_smoothing
            )

            # Backpropagation.
            loss.backward()
            # Apply gradient clipping.
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            # Update learning rate before optimizer.step().
            current_lr: float = self._get_learning_rate(self.global_step + 1)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = current_lr
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.global_step += 1

            # Log training metrics every 100 steps.
            if self.global_step % 100 == 0:
                elapsed: float = time.time() - start_time
                # Avoid exp overflow.
                perplexity: float = math.exp(loss.item()) if loss.item() < 100 else float("inf")
                logger.info(
                    f"Step: {self.global_step}/{self.total_steps}, Loss: {loss.item():.4f}, "
                    f"Perplexity: {perplexity:.2f}, LR: {current_lr:.6f}, Elapsed: {elapsed:.2f}s"
                )

            # Perform validation every 1000 steps.
            if self.global_step % 1000 == 0:
                val_loss: float = self.validate()
                logger.info(f"Validation Loss at step {self.global_step}: {val_loss:.4f}")

            # Save checkpoint if checkpoint interval has passed.
            if time.time() - self.last_checkpoint_time > self.checkpoint_interval:
                self.save_checkpoint()
                self.last_checkpoint_time = time.time()

        logger.info("Training complete.")
