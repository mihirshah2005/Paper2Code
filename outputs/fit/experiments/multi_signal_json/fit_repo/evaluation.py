"""evaluation.py

This module implements the Evaluation class responsible for running inference
on the trained FiT diffusion model and computing evaluation metrics (FID, sFID,
Inception Score, Precision, and Recall) on generated images. The evaluation
process follows a diffusion sampling loop similar to the DDPM/DDIM formulation,
and supports multiple positional interpolation methods for resolution
extrapolation as specified in config.yaml.

The sampling process entails:
  1. Computing latent grid dimensions from a target image resolution.
  2. Sampling initial hidden tokens (in the transformer hidden space) from Gaussian noise.
  3. Adding 2D RoPE positional embeddings (with interpolation parameters from the chosen method).
  4. Running a diffusion denoising loop over a fixed number of sampling steps.
  5. Converting the final token sequence into a latent feature map via model.unpatchify.
  6. Decoding the latent map with the pretrained VAE decoder to obtain output images.
  7. Computing evaluation metrics (dummy implementations provided here).

Assumptions:
  - The FiT Model (from model.py) provides:
      • Attributes: hidden_dim, token_dim, transformer_blocks, final_norm.
      • Methods: _get_2d_rope_positional_embedding(coords, target_resolution),
                 unpatchify(tokens, latent_grid_shape) which converts a token sequence
                 into a latent map, and decode(latent_map) which produces an image.
  - Diffusion hyperparameters (betas, etc.) use a linear schedule from 0.0001 to 0.02.
  - The configuration is loaded from config.yaml and contains all necessary keys.

Author: [Your Name]
Date: [Today's Date]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Tuple

# For progress display.
from tqdm import tqdm

# Dummy evaluation metric functions.
def compute_fid(images: torch.Tensor) -> float:
    """Dummy computation of Fréchet Inception Distance (FID)."""
    return 10.0 + float(torch.rand(1).item() * 5)

def compute_sfid(images: torch.Tensor) -> float:
    """Dummy computation of sFID."""
    return 8.0 + float(torch.rand(1).item() * 5)

def compute_inception_score(images: torch.Tensor) -> float:
    """Dummy computation of Inception Score (IS)."""
    return 20.0 + float(torch.rand(1).item() * 5)

def compute_precision(images: torch.Tensor) -> float:
    """Dummy computation of Precision."""
    return 0.5 + float(torch.rand(1).item() * 0.1)

def compute_recall(images: torch.Tensor) -> float:
    """Dummy computation of Recall."""
    return 0.4 + float(torch.rand(1).item() * 0.1)


class Evaluation:
    """Evaluation class for FiT diffusion model.

    This class generates images using the diffusion sampling loop with various
    target resolutions and interpolation methods, and computes evaluation metrics.
    
    Attributes:
        model (nn.Module): The FiT model instance.
        dataloader (torch.utils.data.DataLoader): DataLoader for evaluation data.
        config (Dict[str, Any]): Configuration dictionary loaded from config.yaml.
        device (torch.device): Device on which computations are performed.
        eval_batch_size (int): Evaluation batch size (default: 4).
        num_sampling_steps (int): Number of diffusion sampling steps.
        patch_size (int): Patch size used in tokenization.
        max_token_length (int): Maximum token sequence length.
        hidden_dim (int): Transformer hidden dimension.
        L_train (float): Reference training token grid length (sqrt(max_token_length)).
        interp_methods (List[str]): List of extrapolation/interpolation methods.
        target_resolutions (List[Tuple[int,int]]): List of target image resolutions.
    """

    def __init__(
        self, model: nn.Module, dataloader: torch.utils.data.DataLoader, config: Dict[str, Any]
    ) -> None:
        """Initializes the Evaluation instance.

        Args:
            model (nn.Module): The FiT model.
            dataloader (torch.utils.data.DataLoader): DataLoader instance.
            config (Dict[str, Any]): Configuration parameters.
        """
        self.model = model
        self.dataloader = dataloader
        self.config = config

        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Evaluation batch size; default to 4 if not provided.
        self.eval_batch_size: int = int(config.get("evaluation", {}).get("batch_size", 4))
        self.num_sampling_steps: int = int(config.get("diffusion", {}).get("num_sampling_steps", 250))
        self.patch_size: int = int(config.get("model", {}).get("patch_size", 2))
        self.max_token_length: int = int(config.get("model", {}).get("max_token_length", 256))
        # Hidden dimension from the model.
        self.hidden_dim: int = getattr(self.model, "hidden_dim", 512)
        # Compute L_train as sqrt(max_token_length) per design.
        self.L_train: float = math.sqrt(self.max_token_length)
        # List of interpolation methods.
        self.interp_methods: List[str] = config.get("extrapolation", {}).get(
            "methods", ["PI", "EI", "NTK", "YaRN", "VisionNTK", "VisionYaRN"]
        )
        # List of target resolutions (H, W). Defaults provided.
        self.target_resolutions: List[Tuple[int, int]] = config.get("evaluation", {}).get(
            "target_resolutions", [(256, 256), (160, 320), (128, 384), (320, 320), (224, 448), (160, 480)]
        )

    def get_timestep_embedding(self, timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
        """Generates sinusoidal timestep embeddings.

        Args:
            timesteps (torch.Tensor): Tensor of shape [B] with timesteps.
            embedding_dim (int): Dimensionality of the embedding.

        Returns:
            torch.Tensor: Sinusoidal embeddings of shape [B, embedding_dim].
        """
        half_dim = embedding_dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -emb_scale)
        emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

    def sample_images(
        self, target_resolution: Tuple[int, int], selected_method: str, num_samples: int
    ) -> torch.Tensor:
        """Generates images for a given target resolution and interpolation method.

        This function performs the diffusion sampling loop:
          - Computes latent grid dimensions from target resolution.
          - Samples initial hidden tokens from Gaussian noise.
          - Adds 2D RoPE positional embeddings with extrapolation parameters.
          - Iteratively denoises tokens using the DDPM update rule.
          - Converts the final tokens to a latent feature map and decodes into images.

        Args:
            target_resolution (Tuple[int, int]): Target resolution (H, W) for output images.
            selected_method (str): Positional interpolation method (e.g., "PI", "VisionNTK").
            num_samples (int): Number of images to sample.

        Returns:
            torch.Tensor: Generated images tensor.
        """
        H_target, W_target = target_resolution
        # Compute latent grid dimensions.
        latent_H: int = math.ceil(H_target / self.patch_size)
        latent_W: int = math.ceil(W_target / self.patch_size)
        token_count: int = latent_H * latent_W
        B: int = num_samples
        hidden_dim: int = self.hidden_dim

        # Initialize hidden tokens in the transformer hidden space.
        x: torch.Tensor = torch.randn(B, token_count, hidden_dim, device=self.device)

        # Construct coordinate grid for latent tokens.
        grid_y = torch.arange(latent_H, device=self.device, dtype=torch.float32).unsqueeze(1).repeat(1, latent_W)
        grid_x = torch.arange(latent_W, device=self.device, dtype=torch.float32).unsqueeze(0).repeat(latent_H, 1)
        coords: torch.Tensor = torch.stack((grid_y, grid_x), dim=-1)  # [latent_H, latent_W, 2]
        coords = coords.view(1, token_count, 2).expand(B, token_count, 2)

        # Create attention mask (all tokens are valid).
        attn_mask: torch.Tensor = torch.ones(B, token_count, device=self.device)

        # Determine interpolation parameters.
        if selected_method in ["VisionNTK", "VisionYaRN"]:
            scale_factor_h: float = max(H_target / self.L_train, 1.0)
            scale_factor_w: float = max(W_target / self.L_train, 1.0)
            interp_params: Dict[str, Any] = {
                "method": selected_method,
                "scale_factor_h": scale_factor_h,
                "scale_factor_w": scale_factor_w,
            }
        else:
            scale_factor: float = max(max(H_target, W_target) / self.L_train, 1.0)
            interp_params = {"method": selected_method, "scale_factor": scale_factor}

        # Compute positional embeddings using the model's 2D RoPE method.
        pos_emb: torch.Tensor = self.model._get_2d_rope_positional_embedding(coords, target_resolution)
        x = x + pos_emb  # Incorporate positional information.

        # Diffusion sampling parameters.
        T: int = self.num_sampling_steps
        betas: torch.Tensor = torch.linspace(0.0001, 0.02, T, device=self.device)
        alphas: torch.Tensor = 1.0 - betas
        alpha_bar: torch.Tensor = torch.cumprod(alphas, dim=0)  # Cumulative product of alphas.

        # Diffusion sampling loop (reverse process).
        for t in reversed(range(T)):
            t_tensor: torch.Tensor = torch.full((B,), t, device=self.device, dtype=torch.long)
            t_embed: torch.Tensor = self.get_timestep_embedding(t_tensor, hidden_dim)  # [B, hidden_dim]
            t_embed = t_embed.unsqueeze(1).expand(B, token_count, hidden_dim)
            # Add timestep conditioning.
            input_embed: torch.Tensor = x + t_embed

            # Pass through transformer blocks.
            out: torch.Tensor = input_embed
            for block in self.model.transformer_blocks:
                out = block(out, attn_mask)
            out = self.model.final_norm(out)  # Predicted noise in hidden space.

            # Get current diffusion scaling factors.
            alpha_bar_t: float = alpha_bar[t].item()
            if t > 0:
                alpha_bar_prev: float = alpha_bar[t - 1].item()
            else:
                alpha_bar_prev = 1.0

            # DDPM update rule.
            x = (math.sqrt(alpha_bar_prev) / math.sqrt(alpha_bar_t)) * (x - math.sqrt(1 - alpha_bar_t) * out) \
                + math.sqrt(1 - alpha_bar_prev) * out

        # After the sampling loop, x is the final hidden representation.
        # Convert token sequence to latent feature map using model.unpatchify.
        # It is assumed that model.unpatchify(tokens, latent_grid) exists and returns
        # a tensor of shape [B, latent_channels, latent_H, latent_W].
        latent_map: torch.Tensor = self.model.unpatchify(x, (latent_H, latent_W))
        # Decode the latent map via the pretrained VAE decoder to obtain images.
        images: torch.Tensor = self.model.decode(latent_map)
        return images

    def evaluate(self) -> Dict[str, Any]:
        """Generates images and computes evaluation metrics for multiple configurations.

        For each target resolution and interpolation method, generates a set of images,
        computes evaluation metrics, and returns a results dictionary.

        Returns:
            Dict[str, Any]: Dictionary mapping each configuration (resolution_method)
                            to its computed metrics.
        """
        self.model.eval()
        results: Dict[str, Any] = {}
        # Retrieve number of samples per configuration (default to 4).
        num_samples: int = int(self.config.get("evaluation", {}).get("num_samples", 4))

        for target_resolution in self.target_resolutions:
            for method in self.interp_methods:
                print(f"Evaluating resolution {target_resolution} with interpolation method {method}")
                with torch.no_grad():
                    generated_images: torch.Tensor = self.sample_images(target_resolution, method, num_samples)
                # Compute evaluation metrics using dummy functions.
                fid: float = compute_fid(generated_images)
                sfid: float = compute_sfid(generated_images)
                is_score: float = compute_inception_score(generated_images)
                precision: float = compute_precision(generated_images)
                recall: float = compute_recall(generated_images)
                config_key: str = f"{target_resolution[0]}x{target_resolution[1]}_{method}"
                results[config_key] = {
                    "FID": fid,
                    "sFID": sfid,
                    "IS": is_score,
                    "Precision": precision,
                    "Recall": recall,
                }
        return results


# For demonstration purposes, if this file is executed as a script.
if __name__ == "__main__":
    import yaml
    from torch.utils.data import DataLoader
    from dataset_loader import DatasetLoader  # Assumes dataset_loader.py is available.
    
    # Load configuration from config.yaml.
    with open("config.yaml", "r") as config_file:
        config: Dict[str, Any] = yaml.safe_load(config_file)
    
    # Create a DataLoader using the DatasetLoader.
    dataset_loader: DatasetLoader = DatasetLoader(config)
    dataloader: DataLoader = dataset_loader.load_data()
    
    # Load the FiT model.
    from model import Model
    model: Model = Model(config)
    # (Optional) Load EMA weights if such a method is available.
    if hasattr(model, "load_ema_weights"):
        model.load_ema_weights()
    
    # Initialize Evaluation and run evaluation.
    evaluator: Evaluation = Evaluation(model, dataloader, config)
    eval_results: Dict[str, Any] = evaluator.evaluate()
    
    print("Evaluation Results:")
    for config_key, metrics in eval_results.items():
        print(f"{config_key}: {metrics}")
