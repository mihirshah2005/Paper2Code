"""trainer.py

This module implements the Trainer class for training the TransformerModel.
It handles the training loop, custom learning rate scheduling, loss computation with label smoothing,
checkpoint saving, and (optional) checkpoint averaging.
"""

import os
import math
import time
import logging
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the masking utility functions from model.py if available; alternatively, reimplement them here.
# For safety we re-use our local versions.
def generate_subsequent_mask(sz: int) -> torch.Tensor:
    """
    Generates a subsequent mask for a sequence of length sz.
    Positions (i, j) are -inf if j > i, and 0 otherwise.
    """
    mask = torch.triu(torch.ones(sz, sz), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask

# ---------------------------
# Label Smoothing Loss
# ---------------------------
class LabelSmoothingLoss(nn.Module):
    """
    Implements cross-entropy loss with label smoothing.
    This loss ignores predictions for padded tokens (specified by ignore_index).
    
    Args:
        smoothing: Label smoothing factor in [0, 1).
        vocab_size: Vocabulary size.
        ignore_index: Token index to ignore (e.g., the padding token).
    """
    def __init__(self, smoothing: float, vocab_size: int, ignore_index: int = 0) -> None:
        super(LabelSmoothingLoss, self).__init__()
        assert 0.0 <= smoothing < 1.0, "Smoothing must be in [0, 1)"
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes the label smoothed loss.
        
        Args:
            logits: Tensor of shape (batch_size, seq_len, vocab_size).
            target: Tensor of shape (batch_size, seq_len) containing target token indices.
            
        Returns:
            Scalar loss.
        """
        batch_size, seq_len, vocab_size = logits.size()
        # Flatten logits and targets.
        logits = logits.view(-1, vocab_size)               # (N, vocab_size)
        target = target.view(-1)                             # (N)
        
        log_probs = torch.log_softmax(logits, dim=-1)         # (N, vocab_size)
        
        # Create the smoothed target distribution.
        true_dist = torch.zeros_like(log_probs)
        # Distribute smoothing equally among all non-ignored classes.
        true_dist.fill_(self.smoothing / (self.vocab_size - 1))
        # For the true token, assign higher probability.
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        # For pad tokens, zero out the distribution.
        true_dist[target == self.ignore_index] = 0
        
        # Compute the loss per token.
        loss = torch.sum(-true_dist * log_probs, dim=-1)
        # Only average over non-pad tokens.
        non_pad_mask = target != self.ignore_index
        if non_pad_mask.sum().item() == 0:
            return torch.tensor(0.0, device=logits.device)
        loss = loss[non_pad_mask].mean()
        return loss

# ---------------------------
# Trainer Class
# ---------------------------
class Trainer:
    """
    The Trainer class manages the full training process for the Transformer model.
    It sets up the optimizer, custom learning rate scheduler, loss function (with label smoothing),
    and handles checkpoint saving/averaging as well as periodic validation.
    
    Data structures and interfaces:
      + __init__(model: TransformerModel, train_loader: DataLoader, val_loader: DataLoader, config: dict)
      + train() : None
      + save_checkpoint() : None
    """
    def __init__(self, model: nn.Module, train_loader: torch.utils.data.DataLoader,
                 val_loader: torch.utils.data.DataLoader, config: Dict[str, Any]) -> None:
        """
        Initializes the Trainer.
        
        Args:
            model: An instance of the TransformerModel.
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            config: Configuration dictionary parsed from config.yaml.
        """
        self.config = config
        self.model = model

        # Device setup: Prefer GPU using CUDA if available.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logging.info(f"Using device: {self.device}")

        # Set padding token index. According to dataset_loader.py the pad token is 0.
        self.pad_token = 0

        # Optimizer hyperparameters (with defaults if missing).
        beta1 = float(self.config.get("training", {}).get("beta1", 0.9))
        beta2 = float(self.config.get("training", {}).get("beta2", 0.98))
        epsilon = float(self.config.get("training", {}).get("epsilon", 1e-9))

        # Retrieve model dimensionality.
        if hasattr(self.model, "d_model"):
            self.d_model = self.model.d_model
        else:
            self.d_model = self.model.shared_embedding.weight.size(1)

        # Warmup steps and total training steps.
        self.warmup_steps = int(self.config.get("training", {}).get("warmup_steps", 4000))
        # Choose total_steps based on model type: use "big" if d_model >= 1024, else "base".
        if self.d_model >= 1024:
            self.total_steps = int(self.config.get("training", {}).get("total_steps", {}).get("big", 300000))
            self.checkpoint_average = int(self.config.get("checkpoint", {}).get("average_checkpoints", {}).get("big", 20))
        else:
            self.total_steps = int(self.config.get("training", {}).get("total_steps", {}).get("base", 100000))
            self.checkpoint_average = int(self.config.get("checkpoint", {}).get("average_checkpoints", {}).get("base", 5))

        # Dropout rate and label smoothing.
        self.dropout_rate = float(self.config.get("training", {}).get("dropout_rate", 0.1))
        self.label_smoothing = float(self.config.get("training", {}).get("label_smoothing", 0.1))

        # Initialize optimizer with an initial learning rate.
        initial_lr = self.d_model ** (-0.5)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=initial_lr,
                                          betas=(beta1, beta2), eps=epsilon)
        self.current_lr = initial_lr

        # Create the loss function with label smoothing, ensuring that padded tokens are ignored.
        # Use the vocabulary size from the shared embedding.
        vocab_size = self.model.shared_embedding.num_embeddings
        self.loss_fn = LabelSmoothingLoss(self.label_smoothing, vocab_size, ignore_index=self.pad_token)

        # Global training step.
        self.global_step = 0

        # Gradient clipping parameter.
        self.max_grad_norm = float(self.config.get("training", {}).get("max_grad_norm", 5.0))

        # Checkpoint settings.
        self.checkpoint_interval_minutes = float(self.config.get("checkpoint", {}).get("checkpoint_interval_minutes", 10))
        self.checkpoint_dir = "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoints: List[str] = []  # List to hold checkpoint file paths.

        # Logging frequency.
        self.log_interval = 100  # Log every 100 steps.

        # Record the last checkpoint time.
        self.last_checkpoint_time = time.time()

        # Store the validation DataLoader.
        self.val_loader = val_loader

        # Set the model to training mode.
        self.model.train()

    def compute_learning_rate(self, step: int) -> float:
        """
        Computes the learning rate following: 
            lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
        
        Args:
            step: Current training step (should be >= 1).
        
        Returns:
            The computed learning rate.
        """
        step = max(step, 1)
        lr = self.d_model ** (-0.5) * min(step ** (-0.5), step * (self.warmup_steps ** (-1.5)))
        return lr

    def update_learning_rate(self) -> None:
        """
        Updates the optimizer's learning rate based on the current global step.
        """
        lr = self.compute_learning_rate(self.global_step)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.current_lr = lr

    def save_checkpoint(self) -> None:
        """
        Saves a checkpoint containing the model state, optimizer state, global step, and configuration.
        """
        checkpoint_state = {
            "global_step": self.global_step,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "config": self.config,
        }
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{self.global_step}.pt")
        try:
            torch.save(checkpoint_state, checkpoint_path)
            self.checkpoints.append(checkpoint_path)
            logging.info(f"Checkpoint saved at step {self.global_step}: {checkpoint_path}")
        except Exception as e:
            logging.error(f"Error saving checkpoint at step {self.global_step}: {e}")

    def average_checkpoints(self) -> Dict[str, torch.Tensor]:
        """
        Averages the state dictionaries of the most recent checkpoints.
        
        Returns:
            A state dictionary with averaged parameter values.
        """
        if not self.checkpoints:
            logging.warning("No checkpoints available for averaging.")
            return {}
        # Select the last N checkpoints based on the configuration.
        selected_ckpts = self.checkpoints[-self.checkpoint_average:]
        avg_state_dict = None
        num_ckpts = len(selected_ckpts)
        for ckpt_path in selected_ckpts:
            ckpt = torch.load(ckpt_path, map_location=self.device)
            state_dict = ckpt["model_state"]
            if avg_state_dict is None:
                avg_state_dict = {k: v.clone().float() for k, v in state_dict.items()}
            else:
                for k, v in state_dict.items():
                    avg_state_dict[k] += v.clone().float()
        for k in avg_state_dict.keys():
            avg_state_dict[k] /= num_ckpts
        logging.info(f"Averaged {num_ckpts} checkpoints.")
        return avg_state_dict

    def validate(self) -> float:
        """
        Runs a validation pass over the validation set and computes the average loss.
        
        Returns:
            Average loss (per non-padded token) over the validation set.
        """
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for batch in self.val_loader:
                src = batch["src"].to(self.device)   # (batch, src_len)
                tgt = batch["tgt"].to(self.device)     # (batch, tgt_len)
                # Prepare target input and output (shifted by one).
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                # Create source padding mask: shape (batch, 1, 1, src_len)
                src_mask = (src == self.pad_token).unsqueeze(1).unsqueeze(2).float()
                src_mask = src_mask.masked_fill(src_mask != 0, float('-inf'))
                # For target: build the subsequent mask.
                tgt_seq_len = tgt_input.size(1)
                subsequent_mask = generate_subsequent_mask(tgt_seq_len).to(self.device)  # (tgt_seq_len, tgt_seq_len)
                # Create target padding mask.
                tgt_pad_mask = (tgt_input == self.pad_token)  # (batch, tgt_seq_len)
                tgt_padding_mask = tgt_pad_mask.unsqueeze(1).expand(-1, tgt_seq_len, -1).float()
                tgt_padding_mask = tgt_padding_mask.masked_fill(tgt_padding_mask != 0, float('-inf'))
                # Combine masks.
                tgt_mask = subsequent_mask.unsqueeze(0) + tgt_padding_mask  # (batch, tgt_seq_len, tgt_seq_len)
                tgt_mask = tgt_mask.unsqueeze(1)  # (batch, 1, tgt_seq_len, tgt_seq_len)
                
                logits = self.model(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask)
                loss = self.loss_fn(logits, tgt_output)
                # Count non-padded tokens.
                non_pad = (tgt_output != self.pad_token).sum().item()
                total_loss += loss.item() * non_pad
                total_tokens += non_pad
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        self.model.train()
        return avg_loss

    def train(self) -> None:
        """
        Runs the full training loop until the total training steps are reached.
        Iterates over training batches, updates the model, logs metrics, performs validation,
        and saves checkpoints periodically.
        """
        start_time = time.time()
        self.model.train()
        train_loader_iter = iter(self.model.training)  # dummy initialization if needed

        # Use our provided train_loader (must be re-iterated in each epoch).
        train_loader_iter = iter(train_loader)

        while self.global_step < self.total_steps:
            try:
                batch = next(train_loader_iter)
            except StopIteration:
                train_loader_iter = iter(train_loader)
                batch = next(train_loader_iter)

            src = batch["src"].to(self.device)  # (batch, src_len)
            tgt = batch["tgt"].to(self.device)  # (batch, tgt_len)

            # Prepare target input (for decoder input) and target output (for loss computation).
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            # Create source padding mask: shape (batch, 1, 1, src_len)
            src_mask = (src == self.pad_token).unsqueeze(1).unsqueeze(2).float()
            src_mask = src_mask.masked_fill(src_mask != 0, float('-inf'))

            # Create target mask: combine subsequent mask with target padding mask.
            tgt_seq_len = tgt_input.size(1)
            subsequent_mask = generate_subsequent_mask(tgt_seq_len).to(self.device)  # (tgt_seq_len, tgt_seq_len)
            tgt_pad_mask = (tgt_input == self.pad_token)  # (batch, tgt_seq_len)
            tgt_padding_mask = tgt_pad_mask.unsqueeze(1).expand(-1, tgt_seq_len, -1).float()
            tgt_padding_mask = tgt_padding_mask.masked_fill(tgt_padding_mask != 0, float('-inf'))
            tgt_mask = subsequent_mask.unsqueeze(0) + tgt_padding_mask  # (batch, tgt_seq_len, tgt_seq_len)
            tgt_mask = tgt_mask.unsqueeze(1)  # (batch, 1, tgt_seq_len, tgt_seq_len)

            # Forward pass.
            logits = self.model(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask)
            # Compute loss using label smoothing and ignoring pad tokens.
            loss = self.loss_fn(logits, tgt_output)

            # Backpropagation.
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Update global step and the learning rate.
            self.global_step += 1
            self.update_learning_rate()

            # Logging.
            if self.global_step % self.log_interval == 0:
                elapsed = time.time() - start_time
                perplexity = math.exp(loss.item())
                logging.info(f"Step: {self.global_step}, Loss: {loss.item():.4f}, "
                             f"Perplexity: {perplexity:.4f}, LR: {self.current_lr:.8f}, "
                             f"Elapsed: {elapsed:.2f} sec")

            # Periodic validation every 1000 steps.
            if self.global_step % 1000 == 0:
                val_loss = self.validate()
                val_perplexity = math.exp(val_loss)
                logging.info(f"Validation at step {self.global_step}: Loss: {val_loss:.4f}, "
                             f"Perplexity: {val_perplexity:.4f}")

            # Checkpoint saving based on elapsed time.
            if (time.time() - self.last_checkpoint_time) >= self.checkpoint_interval_minutes * 60:
                self.save_checkpoint()
                self.last_checkpoint_time = time.time()

        # Final checkpoint after training.
        self.save_checkpoint()
        logging.info("Training complete.")


# If this module is run directly, the following block demonstrates instantiation and training.
if __name__ == "__main__":
    import yaml
    from dataset_loader import DatasetLoader
    from model import TransformerModel
    from torch.utils.data import DataLoader

    # Load configuration from 'config.yaml'
    with open("config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)

    # Initialize the dataset loader for translation (by default, English-German)
    dataset_loader = DatasetLoader(config, task="translation", dataset_name="english_german")
    train_loader, val_loader, test_loader = dataset_loader.load_data()

    # Use vocabulary size from config (fallback default: 37000)
    vocab_size = int(config["data"]["translation"]["vocabulary"].get("english_german", 37000))
    # Initialize the TransformerModel (base model by default).
    model = TransformerModel(config, model_type="base", vocab_size=vocab_size)

    # Initialize the Trainer.
    trainer = Trainer(model, train_loader, val_loader, config)
    trainer.train()
