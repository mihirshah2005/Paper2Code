"""
model.py

This module implements the Model class for the FiT transformer-based diffusion model.
It integrates a pretrained VAE (from huggingface/stabilityai/sd-vae-ft-ema) for encoding/decoding
images, patchifies the latent code into variable-length token sequences (with padding up to a fixed
length), and builds a transformer backbone with 2D Rotary Positional Embedding (RoPE),
masked multi-head self-attention, and a SwiGLU-based feed-forward network. It also supports
various positional interpolation methods for resolution extrapolation.

Author: Your Name
Date: YYYY-MM-DD
"""

import math
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_2d_rope_embedding(height: int, width: int, embed_dim: int, base: float = 10000.0) -> torch.Tensor:
    """
    Compute 2D Rotary Positional Embedding for a grid of given height and width.
    This implementation constructs separate 1D positional embeddings along the height and width
    dimensions (using cosines as a simple approximation) and concatenates them to form a 2D embedding.

    Args:
        height (int): The height of the token grid.
        width (int): The width of the token grid.
        embed_dim (int): Total embedding dimension (must be even).
        base (float, optional): Rotary base constant. Defaults to 10000.0.

    Returns:
        torch.Tensor: Positional embedding of shape (height * width, embed_dim).
    """
    if embed_dim % 2 != 0:
        raise ValueError("Embedding dimension must be even for 2D RoPE.")

    d_half = embed_dim // 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create position indices for height and width
    pos_h = torch.arange(height, dtype=torch.float32, device=device).unsqueeze(1)  # (height, 1)
    pos_w = torch.arange(width, dtype=torch.float32, device=device).unsqueeze(1)   # (width, 1)

    # Compute the inverse frequency for each dimension
    dim_indices = torch.arange(d_half, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (base ** (dim_indices / d_half))  # (d_half,)

    # Compute 1D positional embeddings for height and width using cosine function
    pe_h = torch.cos(pos_h * inv_freq.unsqueeze(0))  # (height, d_half)
    pe_w = torch.cos(pos_w * inv_freq.unsqueeze(0))  # (width, d_half)

    # Expand to 2D grid: each token gets a concatenation of its height and width embeddings.
    pe_h_exp = pe_h.unsqueeze(1).expand(height, width, d_half)  # (height, width, d_half)
    pe_w_exp = pe_w.unsqueeze(0).expand(height, width, d_half)    # (height, width, d_half)
    pos_embedding = torch.cat([pe_h_exp, pe_w_exp], dim=-1)  # (height, width, embed_dim)
    pos_embedding = pos_embedding.reshape(-1, embed_dim)  # (height * width, embed_dim)
    return pos_embedding


def compute_attention_mask(token_mask: torch.Tensor) -> torch.Tensor:
    """
    Compute the attention mask for masked multi-head self-attention.

    Args:
        token_mask (torch.Tensor): Tensor of shape (B, L) with 1 for valid tokens and 0 for padded tokens.

    Returns:
        torch.Tensor: Attention mask of shape (B, 1, 1, L) with -inf for padded tokens and 0 for valid tokens.
    """
    attn_mask = (token_mask == 0).unsqueeze(1).unsqueeze(2)  # Shape: (B, 1, 1, L)
    attn_mask = attn_mask.float() * -1e9
    return attn_mask


class MaskedMHSA(nn.Module):
    """
    Masked Multi-Head Self-Attention module that supports an attention mask to ignore padded tokens.
    """
    def __init__(self, embed_dim: int, num_heads: int) -> None:
        """
        Args:
            embed_dim (int): Embedding dimension.
            num_heads (int): Number of attention heads.
        """
        super(MaskedMHSA, self).__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for masked multi-head self-attention.

        Args:
            x (torch.Tensor): Input tensor of shape (B, L, embed_dim).
            attn_mask (torch.Tensor): Attention mask of shape (B, 1, 1, L).

        Returns:
            torch.Tensor: Output tensor of shape (B, L, embed_dim).
        """
        B, L, _ = x.shape
        q = self.q_proj(x)  # (B, L, embed_dim)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape and transpose for multi-head attention
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, L, head_dim)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B, num_heads, L, L)
        attn_scores = attn_scores + attn_mask  # Apply mask
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, v)  # (B, num_heads, L, head_dim)

        # Reshape back to (B, L, embed_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, self.embed_dim)
        output = self.out_proj(attn_output)
        return output


class SwiGLU_FFN(nn.Module):
    """
    Feed-Forward Network (FFN) module using the SwiGLU activation.
    The formulation is: output = Linear( SiLU(W1(x)) ⊙ (W2(x)) ).
    """
    def __init__(self, embed_dim: int, hidden_dim: int) -> None:
        """
        Args:
            embed_dim (int): Input and output dimension.
            hidden_dim (int): Hidden dimension for the intermediate computations.
        """
        super(SwiGLU_FFN, self).__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the SwiGLU FFN.

        Args:
            x (torch.Tensor): Input tensor of shape (B, L, embed_dim).

        Returns:
            torch.Tensor: Output tensor of shape (B, L, embed_dim).
        """
        act = F.silu(self.fc1(x))
        gate = self.fc2(x)
        x_ffn = act * gate  # Element-wise multiplication
        x_out = self.out_proj(x_ffn)
        return x_out


class TransformerBlock(nn.Module):
    """
    A single transformer block consisting of masked multi-head self-attention and a SwiGLU FFN,
    with pre-layer normalization and residual connections.
    """
    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int) -> None:
        """
        Args:
            embed_dim (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            hidden_dim (int): Hidden dimension for the FFN.
        """
        super(TransformerBlock, self).__init__()
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.attn = MaskedMHSA(embed_dim, num_heads)
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn = SwiGLU_FFN(embed_dim, hidden_dim)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (B, L, embed_dim).
            attn_mask (torch.Tensor): Attention mask of shape (B, 1, 1, L).

        Returns:
            torch.Tensor: Output tensor of shape (B, L, embed_dim).
        """
        # Multi-head self-attention with residual connection
        residual = x
        x_norm = self.attn_norm(x)
        attn_out = self.attn(x_norm, attn_mask)
        x = residual + attn_out

        # Feed-forward network with residual connection
        residual = x
        x_norm = self.ffn_norm(x)
        ffn_out = self.ffn(x_norm)
        x = residual + ffn_out
        return x


class Model(nn.Module):
    """
    The FiT Model class encapsulates the Flexible Vision Transformer for diffusion.
    It integrates a pretrained VAE for encoding images, tokenizes the latent representation via patchification,
    adds 2D RoPE positional embeddings (with optional interpolation for resolution extrapolation),
    and processes tokens through a transformer backbone with Masked MHSA and SwiGLU FFN.
    """
    def __init__(self, params: Dict[str, Any]) -> None:
        """
        Initialize the FiT Model.

        Args:
            params (Dict[str, Any]): Configuration dictionary (typically loaded from config.yaml).
        """
        super(Model, self).__init__()
        # Parse model configuration
        model_config: Dict[str, Any] = params.get("model", {})
        self.patch_size: int = int(model_config.get("patch_size", 2))
        self.max_token_length: int = int(model_config.get("max_token_length", 256))
        transformer_config: Dict[str, Any] = model_config.get("transformer", {})
        self.positional_embedding_method: str = transformer_config.get("positional_embedding", "2D RoPE")
        self.attention_type: str = transformer_config.get("attention", "Masked MHSA")
        self.ffn_type: str = transformer_config.get("ffn", "SwiGLU")

        # Transformer architecture hyperparameters; defaults provided if not in config
        self.embed_dim: int = int(model_config.get("embed_dim", 512))
        self.num_layers: int = int(model_config.get("num_layers", 12))
        self.num_heads: int = int(model_config.get("num_heads", 8))
        self.ffn_hidden_dim: int = int(model_config.get("ffn_hidden_dim", 4 * self.embed_dim))

        # Pretrained VAE configuration
        self.pretrained_vae_path: str = model_config.get("pretrained_vae", "huggingface/stabilityai/sd-vae-ft-ema")
        self.vae = None  # To be loaded via load_pretrained_vae()

        # Assume latent channels from VAE is 4 (as used in Stable Diffusion) and compute token dimensionality.
        latent_channels: int = 4
        token_dim: int = latent_channels * (self.patch_size ** 2)
        self.token_proj = nn.Linear(token_dim, self.embed_dim)
        self.output_proj = nn.Linear(self.embed_dim, token_dim)

        # Stack transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(self.embed_dim, self.num_heads, self.ffn_hidden_dim)
            for _ in range(self.num_layers)
        ])

    def load_pretrained_vae(self, vae_path: str = None) -> None:
        """
        Load the pretrained VAE from the specified path and freeze its parameters.

        Args:
            vae_path (str, optional): Path to the pretrained VAE checkpoint.
                If None, uses the default from the configuration.
        """
        if vae_path is None:
            vae_path = self.pretrained_vae_path
        try:
            from diffusers import AutoencoderKL
        except ImportError as e:
            raise ImportError("The diffusers library is required to load the pretrained VAE. " + str(e))
        self.vae = AutoencoderKL.from_pretrained(vae_path)
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False

    def patchify_and_pad(self, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        """
        Patchify the latent representations and pad token sequences to the fixed max_token_length.

        Args:
            latent (torch.Tensor): Latent tensor of shape (B, C, H, W) from the VAE encoder.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
                - tokens: Tensor of shape (B, max_token_length, token_dim).
                - token_mask: Tensor of shape (B, max_token_length) with 1 for valid tokens and 0 for padded tokens.
                - grid_dims: Tuple (H_p, W_p) representing the height and width of the token grid.
        """
        B, C, H, W = latent.shape
        patch_size = self.patch_size
        token_dim = C * (patch_size ** 2)
        # Use unfold to extract non-overlapping patches
        patches = F.unfold(latent, kernel_size=patch_size, stride=patch_size)  # (B, token_dim, L)
        patches = patches.transpose(1, 2)  # (B, L, token_dim)
        L = patches.shape[1]
        H_p = H // patch_size
        W_p = W // patch_size
        grid_dims = (H_p, W_p)
        # If number of tokens exceeds max_token_length, clip
        if L > self.max_token_length:
            tokens = patches[:, :self.max_token_length, :]
            valid_token_count = self.max_token_length
        else:
            tokens = patches  # (B, L, token_dim)
            valid_token_count = L
            pad_len = self.max_token_length - L
            pad_tensor = torch.zeros(B, pad_len, token_dim, device=latent.device, dtype=latent.dtype)
            tokens = torch.cat([tokens, pad_tensor], dim=1)
        # Create token mask: 1 for valid tokens, 0 for padding
        token_mask = torch.zeros(B, self.max_token_length, device=latent.device, dtype=torch.int64)
        token_mask[:, :valid_token_count] = 1
        return tokens, token_mask, grid_dims

    def compute_positional_embeddings(self, grid_dims: Tuple[int, int]) -> torch.Tensor:
        """
        Compute 2D RoPE positional embeddings for the given token grid dimensions.

        Args:
            grid_dims (Tuple[int, int]): Tuple (H_p, W_p) representing the token grid dimensions.

        Returns:
            torch.Tensor: Positional embeddings of shape (H_p * W_p, embed_dim).
        """
        H_p, W_p = grid_dims
        pos_embeddings = get_2d_rope_embedding(H_p, W_p, self.embed_dim)
        return pos_embeddings

    def interpolate_positional_embedding(
        self,
        base_embedding: torch.Tensor,
        orig_dims: Tuple[int, int],
        target_dims: Tuple[int, int],
        method: str
    ) -> torch.Tensor:
        """
        Interpolate the positional embedding to match a target token grid resolution.

        Args:
            base_embedding (torch.Tensor): Base positional embedding of shape (orig_H * orig_W, embed_dim).
            orig_dims (Tuple[int, int]): Original grid dimensions (orig_H, orig_W).
            target_dims (Tuple[int, int]): Target grid dimensions (target_H, target_W).
            method (str): Interpolation method, one of ["PI", "EI", "NTK", "YaRN", "VisionNTK", "VisionYaRN"].

        Returns:
            torch.Tensor: Interpolated positional embedding of shape (target_H * target_W, embed_dim).
        """
        if method == "PI":
            # Recompute positional embedding with target dimensions using the same 2D RoPE function.
            return get_2d_rope_embedding(target_dims[0], target_dims[1], self.embed_dim)
        elif method == "EI":
            # Use bilinear interpolation on the reshaped positional embedding.
            pe_2d = base_embedding.view(orig_dims[0], orig_dims[1], self.embed_dim).permute(2, 0, 1).unsqueeze(0)
            pe_interp = F.interpolate(pe_2d, size=target_dims, mode='bilinear', align_corners=False)
            pe_interp = pe_interp.squeeze(0).permute(1, 2, 0).reshape(-1, self.embed_dim)
            return pe_interp
        elif method in ["NTK", "YaRN", "VisionNTK", "VisionYaRN"]:
            # Compute scale factor: s = max(max(H_target, W_target) / L_train, 1.0) with L_train = sqrt(max_token_length)
            L_train = math.sqrt(self.max_token_length)
            s = max(max(target_dims[0], target_dims[1]) / L_train, 1.0)
            adjusted_base = 10000.0 / s
            return get_2d_rope_embedding(target_dims[0], target_dims[1], self.embed_dim, base=adjusted_base)
        else:
            # Default: return the base embedding unchanged.
            return base_embedding

    def forward(
        self, 
        x: torch.Tensor, 
        interp_method: str = None, 
        target_dims: Tuple[int, int] = None
    ) -> torch.Tensor:
        """
        Forward pass of the FiT model.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).
            interp_method (str, optional): Positional embedding interpolation method for resolution
                extrapolation (e.g., "PI", "EI", "NTK", "YaRN", "VisionNTK", "VisionYaRN"). Defaults to None.
            target_dims (Tuple[int, int], optional): Target token grid dimensions (target_H, target_W).
                If provided along with interp_method, the positional embeddings are interpolated accordingly.
                Defaults to None.

        Returns:
            torch.Tensor: Output denoised latent tokens of shape (B, max_token_length, token_dim).
        """
        if self.vae is None:
            raise ValueError("Pretrained VAE is not loaded. Call load_pretrained_vae() before forward.")

        # 1. Encode the input images to latent representations using the pretrained VAE.
        with torch.no_grad():
            latent_dist = self.vae.encode(x).latent_dist
            latent = latent_dist.sample()  # Shape: (B, C, H_lat, W_lat)

        # 2. Patchify the latent representation and pad token sequences.
        tokens, token_mask, grid_dims = self.patchify_and_pad(latent)  # tokens: (B, L, token_dim)

        # 3. Project tokens into the transformer embedding space.
        token_embeddings = self.token_proj(tokens)  # (B, L, embed_dim)

        # 4. Compute positional embeddings based on the original grid dimensions.
        pos_embeddings = self.compute_positional_embeddings(grid_dims)  # (L_orig, embed_dim)
        orig_L = grid_dims[0] * grid_dims[1]

        # 5. If interpolation for resolution extrapolation is requested, apply it.
        if target_dims is not None and interp_method is not None:
            pos_embeddings = self.interpolate_positional_embedding(pos_embeddings, grid_dims, target_dims, interp_method)
        else:
            # Ensure positional embeddings match max_token_length: clip or pad if necessary.
            if orig_L > self.max_token_length:
                pos_embeddings = pos_embeddings[:self.max_token_length, :]
            elif orig_L < self.max_token_length:
                pad_len = self.max_token_length - orig_L
                pad_embedding = torch.zeros(pad_len, self.embed_dim, device=pos_embeddings.device, dtype=pos_embeddings.dtype)
                pos_embeddings = torch.cat([pos_embeddings, pad_embedding], dim=0)

        # Expand positional embeddings to batch dimension and add to token embeddings.
        pos_embeddings = pos_embeddings.unsqueeze(0)  # (1, L, embed_dim)
        token_embeddings = token_embeddings + pos_embeddings

        # 6. Compute attention mask from the token mask.
        attn_mask = compute_attention_mask(token_mask)  # (B, 1, 1, L)

        # 7. Process the embeddings through the transformer blocks.
        x_trans = token_embeddings
        for block in self.transformer_blocks:
            x_trans = block(x_trans, attn_mask)

        # 8. Project the transformer output back to the original token dimension.
        output_tokens = self.output_proj(x_trans)  # (B, L, token_dim)
        return output_tokens
