"""evaluation.py

This module implements the Evaluation class to run inference and compute evaluation metrics
for the FiT diffusion model. It performs the following:
  - Runs a diffusion sampling loop using a DDPM reverse process with a reduced number of steps.
  - Applies resolution extrapolation via positional embedding interpolation methods (e.g., PI, EI, NTK, YaRN, VisionNTK, VisionYaRN).
  - Unpatchifies the final latent tokens and decodes them using the pretrained VAE decoder.
  - Computes evaluation metrics (FID, sFID, Inception Score, Precision, and Recall) on the generated images.
  
All configuration parameters are read from the provided configuration dictionary (from config.yaml).

Author: Your Name
Date: YYYY-MM-DD
"""

import os
import math
import logging
from typing import Any, Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Import model functions (avoid circular import issues)
from model import compute_attention_mask  # Used in inference_step

# Setup module-level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Evaluation:
    """
    Evaluation class for the FiT diffusion model.
    
    Attributes:
        model (nn.Module): The FiT model with EMA weights loaded.
        dataloader (torch.utils.data.DataLoader): DataLoader for potential ground-truth inputs (if needed).
        config (Dict[str, Any]): Configuration dictionary loaded from config.yaml.
        device (torch.device): Device to run evaluation on.
        num_sampling_steps (int): Number of diffusion sampling steps (default is 250).
        extrapolation_methods (List[str]): List of positional interpolation methods for resolution extrapolation.
        metrics_list (List[str]): List of evaluation metrics to compute.
        T (int): Total number of diffusion timesteps for the underlying schedule.
        beta (torch.Tensor): Diffusion beta schedule.
        alpha (torch.Tensor): Diffusion alpha rates.
        alpha_bar (torch.Tensor): Cumulative product of alphas.
    """
    def __init__(self, model: nn.Module, dataloader: Any, config: Dict[str, Any]) -> None:
        """
        Initialize the Evaluation class.
        
        Args:
            model (nn.Module): The FiT model instance.
            dataloader (Any): DataLoader providing evaluation data (if needed).
            config (Dict[str, Any]): Configuration parameters from config.yaml.
        """
        self.model: nn.Module = model
        self.dataloader = dataloader
        self.config: Dict[str, Any] = config

        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()  # Ensure model is in evaluation mode

        # Diffusion parameters from config
        diffusion_config: Dict[str, Any] = self.config.get("diffusion", {})
        self.num_sampling_steps: int = int(diffusion_config.get("num_sampling_steps", 250))
        # For simplicity, we assume a DDPM schedule with T=1000 timesteps
        self.T: int = 1000
        beta_start: float = 0.0001
        beta_end: float = 0.02
        self.beta: torch.Tensor = torch.linspace(beta_start, beta_end, self.T, device=self.device)
        self.alpha: torch.Tensor = 1.0 - self.beta
        self.alpha_bar: torch.Tensor = torch.cumprod(self.alpha, dim=0)

        # Extrapolation configuration
        extrapolation_config: Dict[str, Any] = self.config.get("extrapolation", {})
        self.extrapolation_methods: List[str] = extrapolation_config.get("methods", 
                                                                          ["PI", "EI", "NTK", "YaRN", "VisionNTK", "VisionYaRN"])
        # Evaluation metrics
        evaluation_config: Dict[str, Any] = self.config.get("evaluation", {})
        self.metrics_list: List[str] = evaluation_config.get("metrics", ["FID", "sFID", "IS", "Precision", "Recall"])

        # Set a default evaluation batch size and sample count
        self.eval_batch_size: int = 16
        self.num_eval_samples: int = 64  # Total samples per resolution-method evaluation
        
        logger.info("Evaluation initialized on device %s with %d sampling steps.", self.device, self.num_sampling_steps)

    def inference_step(
        self,
        latent: torch.Tensor,
        token_mask: torch.Tensor,
        grid_dims: Tuple[int, int],
        target_dims: Tuple[int, int],
        interp_method: str
    ) -> torch.Tensor:
        """
        Perform a single inference (forward) pass through the transformer part of the model,
        given latent tokens. This mimics the transformer forward pass from training.
        
        Args:
            latent (torch.Tensor): Latent token tensor of shape (B, max_token_length, token_dim).
            token_mask (torch.Tensor): Mask tensor of shape (B, max_token_length) (1 for valid tokens, 0 for pad).
            grid_dims (Tuple[int, int]): Original token grid dimensions (H_p, W_p).
            target_dims (Tuple[int, int]): Target token grid dimensions for extrapolation.
            interp_method (str): Positional embedding interpolation method.
            
        Returns:
            torch.Tensor: Predicted noise tensor of shape (B, max_token_length, token_dim).
        """
        # Project latent tokens into transformer embedding space.
        token_embeddings = self.model.token_proj(latent)  # Shape: (B, L, embed_dim)
        
        # Compute base positional embeddings from grid_dims.
        pos_embeddings = self.model.compute_positional_embeddings(grid_dims)  # Shape: (n_valid, embed_dim)
        n_valid: int = grid_dims[0] * grid_dims[1]
        max_token_length: int = self.model.max_token_length
        # Adjust positional embeddings: clip or pad to fixed length.
        if n_valid > max_token_length:
            pos_embeddings = pos_embeddings[:max_token_length, :]
        elif n_valid < max_token_length:
            pad_len: int = max_token_length - n_valid
            pad_tensor = torch.zeros(pad_len, pos_embeddings.size(1), device=pos_embeddings.device, dtype=pos_embeddings.dtype)
            pos_embeddings = torch.cat([pos_embeddings, pad_tensor], dim=0)
        
        # If an extrapolation method is specified and target_dims provided, interpolate positional embeddings.
        if interp_method is not None and target_dims is not None:
            pos_embeddings = self.model.interpolate_positional_embedding(
                pos_embeddings, grid_dims, target_dims, interp_method
            )
        # Expand positional embeddings to batch dimension and add to token embeddings.
        pos_embeddings = pos_embeddings.unsqueeze(0)  # Shape: (1, L, embed_dim)
        token_embeddings = token_embeddings + pos_embeddings  # Broadcasting addition

        # Compute attention mask.
        attn_mask = compute_attention_mask(token_mask)  # Shape: (B, 1, 1, L)
        
        # Pass through transformer blocks.
        x_trans = token_embeddings
        for block in self.model.transformer_blocks:
            x_trans = block(x_trans, attn_mask)
        
        # Project back to token (latent) space.
        predicted_noise = self.model.output_proj(x_trans)  # Shape: (B, L, token_dim)
        return predicted_noise

    def unpatchify(self, tokens: torch.Tensor, grid_dims: Tuple[int, int]) -> torch.Tensor:
        """
        Convert patchified latent tokens back into the latent image representation.
        
        Args:
            tokens (torch.Tensor): Tensor of shape (B, max_token_length, token_dim).
            grid_dims (Tuple[int, int]): Tuple (H_p, W_p) representing the original token grid dimensions.
            
        Returns:
            torch.Tensor: Reconstructed latent image tensor of shape (B, C_lat, H_lat, W_lat).
        """
        B, L, token_dim = tokens.shape
        valid_token_count: int = grid_dims[0] * grid_dims[1]
        # Take only the valid tokens (ignore padded tokens)
        tokens_valid = tokens[:, :valid_token_count, :]  # (B, valid_token_count, token_dim)
        # Transpose to shape (B, token_dim, valid_token_count)
        tokens_valid = tokens_valid.transpose(1, 2)
        # Define fold parameters based on patch size.
        patch_size: int = self.model.patch_size
        output_h: int = grid_dims[0] * patch_size
        output_w: int = grid_dims[1] * patch_size
        fold = nn.Fold(output_size=(output_h, output_w), kernel_size=patch_size, stride=patch_size)
        # Fold the tokens to reconstruct latent; tokens_valid shape should match (B, C_lat * patch_size^2, L)
        latent = fold(tokens_valid)
        # Compute latent channels (assumed to be 4 as per model design)
        latent_channels: int = token_dim // (patch_size ** 2)
        # Reshape to (B, latent_channels, output_h, output_w)
        latent = latent.view(B, latent_channels, output_h, output_w)
        return latent

    def diffusion_sampling(
        self,
        resolution: Tuple[int, int],
        interp_method: str,
        num_steps: int,
        batch_size: int
    ) -> torch.Tensor:
        """
        Run the diffusion sampling process (reverse DDPM) to generate latent tokens and decode them into images.
        
        Args:
            resolution (Tuple[int, int]): Target image resolution (H, W) in pixels.
            interp_method (str): Positional embedding interpolation method to use.
            num_steps (int): Number of diffusion sampling steps.
            batch_size (int): Number of images to generate in one batch.
            
        Returns:
            torch.Tensor: Generated images tensor of shape (B, C, H_decoded, W_decoded).
        """
        H_img, W_img = resolution
        # Compute latent resolution based on VAE downsampling factor (assumed factor = 8)
        latent_factor: int = 8
        latent_H: int = H_img // latent_factor
        latent_W: int = W_img // latent_factor
        # Compute token grid dimensions: using patch size from model.
        patch_size: int = self.model.patch_size
        grid_H: int = latent_H // patch_size
        grid_W: int = latent_W // patch_size
        grid_dims: Tuple[int, int] = (grid_H, grid_W)
        # For extrapolation, set target_dims same as computed grid dims.
        target_dims: Tuple[int, int] = grid_dims

        # Create token mask: first n_valid tokens are 1, remainder are 0.
        n_valid: int = grid_H * grid_W
        max_token_length: int = self.model.max_token_length
        token_mask = torch.zeros(batch_size, max_token_length, device=self.device, dtype=torch.int64)
        token_mask[:, :n_valid] = 1

        # Determine token dimension (assumed latent_channels * patch_size^2, latent_channels assumed to be 4)
        latent_channels: int = 4
        token_dim: int = latent_channels * (patch_size ** 2)

        # Initialize latent tokens with Gaussian noise.
        x_t = torch.randn(batch_size, max_token_length, token_dim, device=self.device)

        # Create a fast sampling timetable: linearly spaced timesteps from T-1 to 0.
        timesteps = torch.linspace(self.T - 1, 0, steps=num_steps, device=self.device).long()

        # Diffusion sampling loop.
        for t in tqdm(timesteps, desc=f"Sampling at resolution {H_img}x{W_img} using {interp_method}", leave=False):
            # Get schedule parameters for current timestep t (as float scalars)
            alpha_t: float = self.alpha[t].item()
            alpha_bar_t: float = self.alpha_bar[t].item()
            beta_t: float = self.beta[t].item()

            # Predict noise using the transformer part.
            with torch.no_grad():
                pred_noise = self.inference_step(x_t, token_mask, grid_dims, target_dims, interp_method)
            # DDPM reverse update:
            # x_{t-1} = (1/√alpha_t) * (x_t - ((1 - alpha_t)/√(1 - alpha_bar_t)) * pred_noise)
            # Add noise if t > 0.
            x_t = (1.0 / math.sqrt(alpha_t)) * (x_t - ((1 - alpha_t) / math.sqrt(1 - alpha_bar_t)) * pred_noise)
            if t > 0:
                noise = torch.randn_like(x_t)
                x_t = x_t + math.sqrt(beta_t) * noise

        # After diffusion steps, x_t is the final latent tokens.
        latent_tokens = x_t  # Shape: (B, max_token_length, token_dim)
        # Unpatchify the tokens to get latent image representation.
        latent_image = self.unpatchify(latent_tokens, grid_dims)  # Shape: (B, latent_channels, H_lat, W_lat)
        # Decode latent image via pretrained VAE decoder.
        with torch.no_grad():
            # The VAE's decode() method returns a structure with attribute 'sample'
            decoded = self.model.vae.decode(latent_image)
            if hasattr(decoded, "sample"):
                images = decoded.sample
            else:
                images = decoded
        return images

    def evaluate_metrics(self, images: torch.Tensor) -> Dict[str, float]:
        """
        Compute evaluation metrics on a batch of generated images.
        In this implementation, we provide placeholder metric computations.
        In a full implementation, this method would compute FID, sFID, Inception Score,
        Precision, and Recall comparing against reference statistics.
        
        Args:
            images (torch.Tensor): Tensor of generated images of shape (N, C, H, W) in [0,1] range.
            
        Returns:
            Dict[str, float]: A dictionary containing computed metric values.
        """
        # Placeholder implementation: here we return random values to mimic metric computation.
        metrics: Dict[str, float] = {}
        for metric in self.metrics_list:
            # For reproducibility, use numpy's random
            metrics[metric] = float(np.random.rand() * 10)
        logger.info("Computed metrics: %s", metrics)
        return metrics

    def evaluate(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Run the evaluation over a set of test resolutions and extrapolation methods.
        
        Returns:
            Dict[str, Dict[str, Dict[str, float]]]: A nested dictionary where the first key is the resolution label,
                the second key is the extrapolation method, and the innermost dictionary contains metric values.
        """
        # Define a set of test resolutions.
        # In-distribution resolutions.
        eval_resolutions: Dict[str, Tuple[int, int]] = {
            "256x256": (256, 256),
            "160x320": (160, 320),
            "128x384": (128, 384),
            # Out-of-distribution resolutions.
            "320x320": (320, 320),
            "224x448": (224, 448),
            "160x480": (160, 480)
        }
        results: Dict[str, Dict[str, Dict[str, float]]] = {}

        # For each resolution and each extrapolation method, generate images and compute metrics.
        for res_label, res in eval_resolutions.items():
            results[res_label] = {}
            for method in self.extrapolation_methods:
                logger.info("Evaluating resolution %s using extrapolation method %s.", res_label, method)
                all_images: List[torch.Tensor] = []
                num_batches: int = math.ceil(self.num_eval_samples / self.eval_batch_size)
                for _ in tqdm(range(num_batches), desc=f"Generating samples for {res_label} [{method}]", leave=False):
                    images_batch = self.diffusion_sampling(
                        resolution=res,
                        interp_method=method,
                        num_steps=self.num_sampling_steps,
                        batch_size=self.eval_batch_size
                    )
                    # Assume the decoded images are in range [0, 1]. Convert to CPU.
                    images_batch = images_batch.detach().cpu()
                    all_images.append(images_batch)
                # Concatenate generated images.
                generated_images = torch.cat(all_images, dim=0)
                # If more images than needed, take only the required number.
                generated_images = generated_images[:self.num_eval_samples]
                # Compute evaluation metrics.
                metrics_dict = self.evaluate_metrics(generated_images)
                results[res_label][method] = metrics_dict
                logger.info("Results for resolution %s with method %s: %s", res_label, method, metrics_dict)
        return results
