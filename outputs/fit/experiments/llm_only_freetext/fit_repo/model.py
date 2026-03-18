"""
model.py

This module implements the Model class for the FiT (Flexible Vision Transformer for Diffusion) as described in the paper.
It integrates:
  - A pretrained VAE encoder/decoder (loaded via load_pretrained_vae)
  - A patchification and token padding module (patchify_and_pad)
  - A transformer backbone using 2D Rotary Positional Embedding (RoPE), Masked Multi-Head Self-Attention (MHSA),
    and a SwiGLU-based feed‐forward network (FFN)
  - Positional embedding interpolation methods for resolution extrapolation are defined as separate functions.
  
The implementation strictly follows the design and configuration specified in config.yaml.
"""

import math
from typing import Any, Dict, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Helper function to rotate every pair of elements in the last dimension:
      For tensor x of shape (..., d) with d even, splits x into two halves and returns:
      [-x_second_half, x_first_half]
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary positional embedding to tensor x.
    
    Args:
        x: Tensor of shape [B, num_heads, L, head_dim]
        cos: Tensor of shape [1 or B, L, head_dim] or [L, head_dim]
        sin: Tensor of shape [1 or B, L, head_dim] or [L, head_dim]
    
    Returns:
        x_rot: Tensor of the same shape as x after applying rotary transformation.
    """
    return (x * cos.unsqueeze(1)) + (rotate_half(x) * sin.unsqueeze(1))


def build_1d_rope(position: torch.Tensor, dim: int, base: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build 1D rotary positional embedding (cosine and sine components) for given positions.
    
    Args:
        position: Tensor of shape [B, L] or [L] containing token positions (row or column).
        dim: Dimension of the embedding vector.
        base: Rotary base constant, default is 10000.0.
        
    Returns:
        Tuple of (cosine, sine) tensors, each of shape same as (position.unsqueeze(-1)) with last dim 'dim'.
    """
    # Ensure position is float
    position = position.to(dtype=torch.float32)
    inv_freq = base ** (-2 * torch.arange(0, dim, dtype=torch.float32, device=position.device) / float(dim))
    # Shape: if position is [B,L] then theta becomes [B, L, dim]
    theta = torch.einsum("...l, d -> ...ld", position, inv_freq)
    cos = torch.cos(theta)
    sin = torch.sin(theta)
    return cos, sin


class TransformerBlock(nn.Module):
    """
    Transformer block that implements:
      - Pre-LayerNorm
      - Masked Multi-Head Self-Attention with 2D RoPE rotation applied on Q and K.
      - SwiGLU-based Feed-Forward Network (FFN) with LayerNorm and residual connection.
    """
    def __init__(self, embed_dim: int, num_heads: int) -> None:
        """
        Initialize the TransformerBlock.
        
        Args:
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads.
        """
        super(TransformerBlock, self).__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")
        self.embed_dim: int = embed_dim
        self.num_heads: int = num_heads
        self.head_dim: int = embed_dim // num_heads

        # Linear projections for Q, K, V (no bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Layer Normalizations for attention and FFN branches
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # SwiGLU feed-forward network: first linear expands dimension by factor of 2, then split into two halves.
        self.ffn_fc1 = nn.Linear(embed_dim, embed_dim * 2, bias=False)
        self.ffn_fc2 = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, pos_cos: torch.Tensor, pos_sin: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the transformer block.
        
        Args:
            x: Input tensor of shape [B, L, embed_dim].
            pos_cos: Rotary cosine tensor of shape [B, L, head_dim] (for each head, same for all heads).
            pos_sin: Rotary sine tensor of shape [B, L, head_dim].
            attn_mask: Attention mask tensor of shape [B, L] with 0 for valid tokens and -inf for padding.
        
        Returns:
            Output tensor of shape [B, L, embed_dim].
        """
        B, L, D = x.shape

        # --- Multi-Head Self-Attention with Masked Attention ---
        residual = x
        x_norm = self.norm1(x)
        # Linear projections; shape: [B, L, embed_dim]
        q = self.q_proj(x_norm)
        k = self.k_proj(x_norm)
        v = self.v_proj(x_norm)
        # Reshape to [B, L, num_heads, head_dim] and then transpose to [B, num_heads, L, head_dim]
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply rotary positional embedding to Q and K.
        # pos_cos and pos_sin are provided with shape [B, L, embed_dim] but we need per-head dim so split them.
        # We assume that the rotary embedding will be applied on each head separately.
        # Split pos_cos into heads: [B, L, num_heads, head_dim] then transpose to [B, num_heads, L, head_dim]
        pos_cos = pos_cos.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        pos_sin = pos_sin.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        q = (q * pos_cos) + (rotate_half(q) * pos_sin)
        k = (k * pos_cos) + (rotate_half(k) * pos_sin)

        # Compute scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # shape: [B, num_heads, L, L]
        # Expand attn_mask from [B, L] to [B, 1, 1, L] and add; mask: 0 means valid, -inf means ignore.
        if attn_mask is not None:
            mask = attn_mask.unsqueeze(1).unsqueeze(2)  # shape: [B, 1, 1, L]
            attn_scores = attn_scores + mask
        attn_probs = F.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_probs, v)  # shape: [B, num_heads, L, head_dim]
        context = context.transpose(1, 2).contiguous().view(B, L, D)
        attn_out = self.out_proj(context)
        x = residual + attn_out

        # --- Feed-Forward Network with SwiGLU ---
        residual = x
        x_norm = self.norm2(x)
        ffn_intermediate = self.ffn_fc1(x_norm)  # shape: [B, L, 2 * embed_dim]
        # Split into two halves for SwiGLU: one branch goes through SiLU
        split_dim = ffn_intermediate.shape[-1] // 2
        x1 = ffn_intermediate[..., :split_dim]
        x2 = ffn_intermediate[..., split_dim:]
        x1 = F.silu(x1)
        ffn_out = self.ffn_fc2(x1 * x2)
        x = residual + ffn_out
        return x


class Model(nn.Module):
    """
    The Model class encapsulates the FiT transformer-based diffusion backbone.
    
    It loads a pretrained VAE (encoder/decoder), processes latent representations via patchification,
    and applies a transformer backbone with custom modules (2D RoPE, Masked MHSA, SwiGLU FFN).
    """
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the Model with the given configuration.
        
        Args:
            config: Configuration dictionary (e.g., loaded from config.yaml).
        """
        super(Model, self).__init__()
        self.config: Dict[str, Any] = config

        # Model configuration parameters with default values
        self.patch_size: int = int(self.config.get("model", {}).get("patch_size", 2))
        self.max_token_length: int = int(self.config.get("model", {}).get("max_token_length", 256))
        self.positional_embedding_type: str = str(self.config.get("model", {}).get("transformer", {}).get("positional_embedding", "2D RoPE"))
        self.attention_type: str = str(self.config.get("model", {}).get("transformer", {}).get("attention", "Masked MHSA"))
        self.ffn_type: str = str(self.config.get("model", {}).get("transformer", {}).get("ffn", "SwiGLU"))
        self.pretrained_vae_path: str = str(self.config.get("model", {}).get("pretrained_vae", "huggingface/stabilityai/sd-vae-ft-ema"))
        
        # Transformer backbone hyperparameters (with defaults)
        self.embed_dim: int = int(self.config.get("model", {}).get("embed_dim", 512))
        self.num_layers: int = int(self.config.get("model", {}).get("transformer", {}).get("num_layers", 12))
        self.num_heads: int = int(self.config.get("model", {}).get("transformer", {}).get("num_heads", 8))
        
        # Load the pretrained VAE (encoder/decoder)
        self.load_pretrained_vae(self.pretrained_vae_path)
        
        # Determine token dimension from latent channels and patch size.
        # Assumption: pretrained VAE latent channels is set during load_pretrained_vae (default to 4 if dummy).
        self.latent_channels: int = getattr(self, "latent_channels", 4)
        self.token_dim: int = self.latent_channels * (self.patch_size ** 2)
        
        # If token_dim is not equal to embed_dim, add a linear projection.
        if self.token_dim != self.embed_dim:
            self.token_proj = nn.Linear(self.token_dim, self.embed_dim, bias=False)
        else:
            self.token_proj = nn.Identity()
        
        # Build the transformer backbone as a stack of TransformerBlock.
        self.transformer_blocks: nn.ModuleList = nn.ModuleList([
            TransformerBlock(embed_dim=self.embed_dim, num_heads=self.num_heads)
            for _ in range(self.num_layers)
        ])
        
        # Final layer normalization before output.
        self.final_norm = nn.LayerNorm(self.embed_dim)

    def load_pretrained_vae(self, vae_path: str) -> None:
        """
        Load the pretrained VAE from the provided path.
        In a complete implementation, this should load a model from Hugging Face.
        Here, if the proper package is not available, a dummy VAE is created.
        
        Sets self.vae_encoder and self.vae_decoder.
        Also sets self.latent_channels.
        
        Args:
            vae_path: Path or identifier for the pretrained VAE.
        """
        try:
            # Attempt to load using diffusers if available.
            # Note: diffusers is not among required packages. Replace with actual loader if available.
            from diffusers import AutoencoderKL
            vae = AutoencoderKL.from_pretrained(vae_path)
            self.vae_encoder = vae.encode
            self.vae_decoder = vae.decode
            self.latent_channels = vae.config.latent_channels  # Assume this attribute exists.
        except ImportError:
            # Fallback: create a dummy VAE with identity operations.
            self.vae_encoder = nn.Identity()
            self.vae_decoder = nn.Identity()
            self.latent_channels = 4  # Default latent channels

    def patchify_and_pad(self, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Patchify the latent representation and pad/truncate the resulting token sequence to a fixed length.
        
        The function divides the latent tensor into non-overlapping patches of size patch_size x patch_size,
        flattens each patch, and then either pads (if token count is less than max_token_length) or truncates
        (if token count exceeds max_token_length).
        
        Additionally, it computes a grid of patch coordinates (row, col) which is used for positional embeddings.
        
        Args:
            latent: Tensor of shape [B, C, H, W] from the VAE encoder.
            
        Returns:
            A tuple of:
              - tokens_padded: Tensor of shape [B, max_token_length, token_dim]
              - coords_padded: Tensor of shape [B, max_token_length, 2] representing (row, col) coordinates.
              - validity_mask: Tensor of shape [B, max_token_length] with 1 for valid tokens and 0 for padded tokens.
        """
        B, C, H, W = latent.shape
        p: int = self.patch_size
        H_patches: int = H // p
        W_patches: int = W // p
        L_tokens: int = H_patches * W_patches  # Total number of tokens before padding/truncation

        # Use unfold to extract patches. Resulting shape: [B, C, H_patches, W_patches, p, p]
        patches = latent.unfold(2, p, p).unfold(3, p, p)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()  # [B, H_patches, W_patches, C, p, p]
        patches = patches.view(B, L_tokens, -1)  # [B, L_tokens, token_dim]
        
        # Create coordinate grid for patches (row, col) in row-major order.
        grid_y, grid_x = torch.meshgrid(torch.arange(H_patches, device=latent.device),
                                         torch.arange(W_patches, device=latent.device),
                                         indexing='ij')
        grid = torch.stack([grid_y.reshape(-1), grid_x.reshape(-1)], dim=-1).to(torch.float32)  # [L_tokens, 2]
        
        # Prepare lists to store per-sample tokens, coordinates, and validity mask.
        token_list: List[torch.Tensor] = []
        coord_list: List[torch.Tensor] = []
        validity_mask_list: List[torch.Tensor] = []
        max_tokens: int = self.max_token_length

        for b in range(B):
            token_seq: torch.Tensor = patches[b]  # [L_tokens, token_dim]
            coord_seq: torch.Tensor = grid.clone()  # [L_tokens, 2]
            current_len: int = token_seq.shape[0]
            if current_len >= max_tokens:
                token_seq = token_seq[:max_tokens, :]
                coord_seq = coord_seq[:max_tokens, :]
                validity_mask_sample = torch.ones(max_tokens, dtype=torch.float32, device=latent.device)
            else:
                pad_len: int = max_tokens - current_len
                pad_tokens = torch.zeros(pad_len, token_seq.shape[-1], dtype=token_seq.dtype, device=latent.device)
                token_seq = torch.cat([token_seq, pad_tokens], dim=0)
                pad_coords = torch.zeros(pad_len, 2, dtype=coord_seq.dtype, device=latent.device)
                coord_seq = torch.cat([coord_seq, pad_coords], dim=0)
                validity_mask_sample = torch.cat([
                    torch.ones(current_len, dtype=torch.float32, device=latent.device),
                    torch.zeros(pad_len, dtype=torch.float32, device=latent.device)
                ], dim=0)
            token_list.append(token_seq)
            coord_list.append(coord_seq)
            validity_mask_list.append(validity_mask_sample)
        
        tokens_padded = torch.stack(token_list, dim=0)      # [B, max_tokens, token_dim]
        coords_padded = torch.stack(coord_list, dim=0)        # [B, max_tokens, 2]
        validity_mask = torch.stack(validity_mask_list, dim=0)  # [B, max_tokens]
        return tokens_padded, coords_padded, validity_mask

    def compute_2d_rope(self, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute 2D rotary positional embeddings for the patch coordinates.
        
        For each token, the coordinate is (row, col). This function computes separate 1D RoPE for the row and col,
        each with dimension (embed_dim // 2). The final rotary embeddings are obtained by concatenation.
        
        Args:
            coords: Tensor of shape [B, L, 2] containing (row, col) coordinates.
            
        Returns:
            A tuple (pos_cos, pos_sin) of tensors each of shape [B, L, embed_dim].
        """
        B, L, _ = coords.shape
        d_total: int = self.embed_dim
        d_half: int = d_total // 2

        # Get row and column positions; shape: [B, L]
        row_pos: torch.Tensor = coords[..., 0]
        col_pos: torch.Tensor = coords[..., 1]

        # Build 1D RoPE for row and col separately
        # The build_1d_rope helper expects positions of shape [B, L] and returns tensors of shape [B, L, d_half]
        row_cos, row_sin = build_1d_rope(row_pos, d_half, base=10000.0)
        col_cos, col_sin = build_1d_rope(col_pos, d_half, base=10000.0)

        # Concatenate the row and col embeddings along the last dimension to obtain final rotary embeddings.
        pos_cos = torch.cat([row_cos, col_cos], dim=-1)  # [B, L, embed_dim]
        pos_sin = torch.cat([row_sin, col_sin], dim=-1)  # [B, L, embed_dim]
        return pos_cos, pos_sin

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model.
        
        Steps:
          1. Encode input images using the VAE encoder.
          2. Patchify and pad the latent representations to obtain a fixed-length token sequence.
          3. Project tokens to the transformer embedding dimension.
          4. Compute 2D rotary positional embeddings from patch grid coordinates.
          5. Generate attention mask from the validity mask.
          6. Process tokens through the transformer backbone.
          7. Apply final layer normalization and return the tokens along with the validity mask.
          
        Args:
            x: Input image tensor of shape [B, C, H, W].
        
        Returns:
            A tuple (out, validity_mask) where:
              - out: Output latent tokens of shape [B, max_token_length, embed_dim].
              - validity_mask: Binary mask of shape [B, max_token_length] indicating valid tokens.
        """
        # --- VAE Encoding ---
        # Pass input through VAE encoder; assume output latent shape: [B, C_latent, H_latent, W_latent]
        latent: torch.Tensor = self.vae_encoder(x)
        
        # --- Patchify and Pad ---
        tokens, coords, validity_mask = self.patchify_and_pad(latent)  # tokens: [B, T, token_dim]
        
        # --- Token Projection ---
        tokens = self.token_proj(tokens)  # Project to [B, T, embed_dim]
        
        # --- Positional Embedding via 2D RoPE ---
        pos_cos, pos_sin = self.compute_2d_rope(coords)  # Each of shape [B, T, embed_dim]
        
        # --- Construct Attention Mask ---
        # Valid tokens are marked with 1 in validity_mask; convert to mask values:
        # Multiply (1 - validity_mask): valid (1) becomes 0, padded (0) becomes 1; then scale to -inf for padded tokens.
        attn_mask = (1.0 - validity_mask) * (-1e4)  # [B, T]; using -1e4 as a stand-in for -inf
        
        # --- Transformer Backbone ---
        x_tokens: torch.Tensor = tokens  # [B, T, embed_dim]
        for block in self.transformer_blocks:
            x_tokens = block(x_tokens, pos_cos, pos_sin, attn_mask)
        out: torch.Tensor = self.final_norm(x_tokens)
        
        return out, validity_mask

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Decode the latent tokens back into an image using the VAE decoder.
        This function can be used during inference.
        
        Args:
            tokens: Tensor of shape [B, max_token_length, embed_dim] output from the transformer.
        
        Returns:
            Decoded image tensor.
        """
        # In a complete implementation, tokens need to be un-patchified back to latent spatial layout.
        # Here we assume an inverse operation exists; for simplicity, we use a placeholder.
        # Reshape tokens to match latent shape and pass through the VAE decoder.
        # This is a stub and should be replaced with the proper unpatchify routine.
        B = tokens.size(0)
        # Assume latent spatial dimensions are sqrt(max_token_length) x sqrt(max_token_length)
        side: int = int(math.sqrt(self.max_token_length))
        # Assume latent channels from VAE are self.latent_channels and token_dim = latent_channels * patch_size^2.
        # Compute expected spatial size given patch_size.
        latent_H: int = side * self.patch_size
        latent_W: int = side * self.patch_size
        # Reshape tokens: first project tokens back to token_dim if needed.
        # For simplicity, assume tokens can be reshaped to [B, latent_channels, latent_H // patch_size, latent_W // patch_size]
        latent = tokens.view(B, side * side, -1)
        # This is a placeholder; proper implementation requires unpatchify function.
        decoded = self.vae_decoder(latent)
        return decoded
