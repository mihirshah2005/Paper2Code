#!/usr/bin/env python3
"""
trainer.py

This module implements the Trainer class which manages the training loop for the
Transformer model. It handles forward and backward passes, learning rate scheduling,
checkpoint saving and averaging, and periodic validation. The training process
utilizes the Adam optimizer with a custom learning rate scheduler as described
in "Attention Is All You Need".

Author: [Your Name]
Date: [Date]
"""

import os
import math
import time
import logging
from typing import Dict, Any, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

# Configure module-level logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LabelSmoothingLoss(nn.Module):
    """
    Implements Label Smoothing Loss.

    Instead of a one-hot target, the target distribution is smoothed so that
    the correct class has confidence (1 - label_smoothing) and each other class
    gets label_smoothing / (num_classes - 1). This loss is computed per token,
    and then averaged over all tokens (ignoring pad tokens).
    """
    def __init__(self, label_smoothing: float, tgt_vocab_size: int, ignore_index: int = 0) -> None:
        super(LabelSmoothingLoss, self).__init__()
        assert 0.0 <= label_smoothing < 1.0, "label_smoothing must be in [0, 1)"
        self.label_smoothing = label_smoothing
        self.tgt_vocab_size = tgt_vocab_size
        self.ignore_index = ignore_index
        self.confidence = 1.0 - label_smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Logits tensor of shape [N, num_classes].
            target: Ground truth indices tensor of shape [N].
        Returns:
            The averaged smoothed loss.
        """
        # Compute log probabilities.
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            # Initialize smoothed labels with uniform distribution
            true_dist = torch.full_like(pred, self.label_smoothing / (self.tgt_vocab_size - 1))
            # Scatter the confidence value to the true class indices.
            target = target.unsqueeze(1)
            true_dist.scatter_(1, target, self.confidence)
            # Zero out the distribution for pad tokens.
            true_dist.masked_fill_((target == self.ignore_index), 0)
        loss = torch.mean(torch.sum(-true_dist * pred, dim=-1))
        return loss


class Trainer:
    """
    Trainer for the Transformer model.

    Attributes:
        model: The TransformerModel instance.
        train_loader: PyTorch DataLoader for training data.
        val_loader: PyTorch DataLoader for validation data.
        config: Configuration dictionary (parsed from config.yaml).
    """
    def __init__(self, model: torch.nn.Module,
                 train_loader: torch.utils.data.DataLoader,
                 val_loader: torch.utils.data.DataLoader,
                 config: Dict[str, Any]) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # Set device and move model to device.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.global_step: int = 0

        # Determine model configuration (base or big) and extract d_model and total_steps.
        model_config: Dict[str, Any] = self.config.get("model", {})
        if "base" in model_config and model_config["base"]:
            self.model_type: str = "base"
            self.d_model: int = int(model_config["base"].get("d_model", 512))
            self.total_steps: int = int(self.config["training"]["total_steps"].get("base_model", 100000))
        else:
            self.model_type = "big"
            self.d_model = int(model_config["big"].get("d_model", 1024))
            self.total_steps = int(self.config["training"]["total_steps"].get("big_model", 300000))

        self.warmup_steps: int = int(self.config["training"].get("warmup_steps", 4000))
        self.label_smoothing: float = float(self.config["training"].get("label_smoothing", 0.1))
        self.dropout_rate: float = float(self.config["training"].get("dropout_rate", 0.1))
        self.beta1: float = float(self.config["training"].get("beta1", 0.9))
        self.beta2: float = float(self.config["training"].get("beta2", 0.98))
        self.epsilon: float = float(self.config["training"].get("epsilon", 1e-9))
        
        # Initialize the Adam optimizer with base lr 1.0; 
        # the actual learning rate is managed by the scheduler.
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=1.0,
            betas=(self.beta1, self.beta2),
            eps=self.epsilon
        )
        # Define custom learning rate scheduler according to the paper:
        # lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
        self.scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: (self.d_model ** -0.5) *
                                     min((max(step, 1)) ** (-0.5),
                                         (max(step, 1)) * (self.warmup_steps ** -1.5))
        )

        # Get target vocabulary size from the model (assumed to be set in model.tgt_vocab_size).
        self.tgt_vocab_size: int = getattr(self.model, "tgt_vocab_size", 0)
        if self.tgt_vocab_size <= 0:
            raise ValueError("Target vocabulary size must be defined in the model.")
        self.criterion = LabelSmoothingLoss(self.label_smoothing, self.tgt_vocab_size, ignore_index=0)

        # For checkpoint management.
        self.last_checkpoint_time: float = time.monotonic()
        self.checkpoint_paths: List[str] = []  # Stores paths of saved checkpoints.
        checkpoint_config = self.config.get("checkpoint", {})
        self.checkpoint_interval_seconds: float = float(checkpoint_config.get("checkpoint_interval_minutes", 10)) * 60

    def train(self) -> None:
        """
        Runs the training loop until the total number of training steps is reached.
        The loop handles DataLoader iterator exhaustion, loss computation with label smoothing,
        backward and optimizer steps, learning rate scheduling, periodic validation, and checkpoint saving.
        """
        self.model.train()
        train_iter = iter(self.train_loader)
        while self.global_step < self.total_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)
            
            # Move batch data to device. For translation tasks, expect keys "source" and "target".
            if "source" in batch and "target" in batch:
                src = batch["source"].to(self.device)
                tgt = batch["target"].to(self.device)
            # For parsing tasks, use "sentence" for both src and tgt.
            elif "sentence" in batch:
                src = batch["sentence"].to(self.device)
                tgt = batch["sentence"].to(self.device)
            else:
                raise ValueError("Batch does not contain expected keys for training.")
            
            # Forward pass: compute logits.
            logits = self.model(src, tgt)
            B, seq_len, vocab_size = logits.size()
            logits_flat = logits.view(-1, vocab_size)
            tgt_flat = tgt.view(-1)
            
            # Compute loss using label smoothing.
            loss = self.criterion(logits_flat, tgt_flat)
            loss_value = loss.item()
            
            # Backward pass and parameter update.
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            self.global_step += 1

            # Logging: every 50 training steps.
            if self.global_step % 50 == 0:
                current_lr = self.optimizer.param_groups[0]["lr"]
                try:
                    perplexity = math.exp(loss_value)
                except OverflowError:
                    perplexity = float("inf")
                logger.info(f"Step: {self.global_step}, Loss: {loss_value:.4f}, "
                            f"LR: {current_lr:.6f}, Perplexity: {perplexity:.2f}")
            
            # Trigger validation every 1000 steps.
            if self.global_step % 1000 == 0:
                self.validate()
            
            # Checkpoint saving based on elapsed time.
            current_time = time.monotonic()
            if current_time - self.last_checkpoint_time >= self.checkpoint_interval_seconds:
                self.save_checkpoint()
                self.last_checkpoint_time = current_time

        # End of training: save final checkpoint and average checkpoints.
        self.save_checkpoint()
        self.average_checkpoints()

    def validate(self) -> None:
        """
        Evaluates the model on the validation set and logs the average loss and perplexity.
        """
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for batch in self.val_loader:
                if "source" in batch and "target" in batch:
                    src = batch["source"].to(self.device)
                    tgt = batch["target"].to(self.device)
                elif "sentence" in batch:
                    src = batch["sentence"].to(self.device)
                    tgt = batch["sentence"].to(self.device)
                else:
                    raise ValueError("Validation batch does not contain expected keys.")
                
                logits = self.model(src, tgt)
                B, seq_len, vocab_size = logits.size()
                logits_flat = logits.view(-1, vocab_size)
                tgt_flat = tgt.view(-1)
                loss = self.criterion(logits_flat, tgt_flat)
                total_loss += loss.item() * tgt_flat.size(0)
                total_tokens += tgt_flat.size(0)
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
        try:
            perplexity = math.exp(avg_loss)
        except OverflowError:
            perplexity = float("inf")
        logger.info(f"Validation - Global Step: {self.global_step}, Avg Loss: {avg_loss:.4f}, "
                    f"Perplexity: {perplexity:.2f}")
        self.model.train()

    def save_checkpoint(self) -> None:
        """
        Saves a checkpoint that includes the model state_dict, optimizer state_dict, and current global step.
        The checkpoint filename includes the current training step for clarity.
        """
        checkpoint = {
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        checkpoint_filename = f"checkpoint_step_{self.global_step}.pt"
        torch.save(checkpoint, checkpoint_filename)
        logger.info(f"Checkpoint saved: {checkpoint_filename}")
        self.checkpoint_paths.append(checkpoint_filename)

    def average_checkpoints(self) -> None:
        """
        Averages the model parameters from the last N checkpoints (where N is defined in the config)
        and saves the averaged state_dict as a new checkpoint. Only model parameters are averaged;
        the optimizer state is not combined.
        """
        checkpoint_config = self.config.get("checkpoint", {})
        if self.model_type == "base":
            avg_count = int(checkpoint_config.get("average_checkpoints", {}).get("base", 5))
        else:
            avg_count = int(checkpoint_config.get("average_checkpoints", {}).get("big", 20))
        
        if not self.checkpoint_paths:
            logger.warning("No checkpoints available for averaging.")
            return

        selected_paths = self.checkpoint_paths[-avg_count:]
        avg_state_dict = None
        for path in selected_paths:
            checkpoint = torch.load(path, map_location=self.device)
            state_dict = checkpoint["model_state_dict"]
            if avg_state_dict is None:
                avg_state_dict = {k: v.clone().float() for k, v in state_dict.items()}
            else:
                for k, v in state_dict.items():
                    avg_state_dict[k] += v.clone().float()
        # Compute element-wise average.
        for k in avg_state_dict:
            avg_state_dict[k] /= len(selected_paths)

        averaged_checkpoint = {
            "global_step": self.global_step,
            "model_state_dict": avg_state_dict,
        }
        avg_filename = f"averaged_checkpoint_step_{self.global_step}.pt"
        torch.save(averaged_checkpoint, avg_filename)
        logger.info(f"Averaged checkpoint saved: {avg_filename}")
