#!/usr/bin/env python3
"""
model.py

This module implements the TransformerModel along with its helper modules:
  - PositionalEncoding: Implements fixed sinusoidal positional encoding.
  - MultiHeadAttention: Implements scaled dot-product attention with multiple heads.
  - PositionwiseFeedForward: Implements a two-layer feed-forward network.
  - EncoderLayer and Encoder: Implements the encoder stack.
  - DecoderLayer and Decoder: Implements the decoder stack.
  - TransformerModel: The complete encoder–decoder model with optional weight tying.

The implementation follows "Attention Is All You Need" with configuration options
driven by an external config.yaml file. The model supports conditional weight tying,
and correct masking for both encoder and decoder self-/encoder–decoder attention.

Author: [Your Name]
Date: [Date]
"""

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    Computes fixed sinusoidal positional encodings.
    
    PE(pos, 2i) = sin(pos / (10000^(2i/d_model)))
    PE(pos, 2i+1) = cos(pos / (10000^(2i/d_model)))
    
    The encoding is dropout applied before addition.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create constant 'pe' matrix with shape [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Shape becomes [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        # Register as buffer so it is saved in the state dictionary
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [B, seq_len, d_model]
        Returns:
            Tensor of shape [B, seq_len, d_model] with positional encodings added.
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-head scaled dot-product attention.
    
    The module projects queries, keys, and values h times in parallel,
    computes scaled dot-product attention on each, concatenates the outputs,
    and runs a final linear projection.
    """
    def __init__(self, d_model: int, num_heads: int, d_k: int, d_v: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v
        self.linear_q = nn.Linear(d_model, num_heads * d_k)
        self.linear_k = nn.Linear(d_model, num_heads * d_k)
        self.linear_v = nn.Linear(d_model, num_heads * d_v)
        self.linear_out = nn.Linear(num_heads * d_v, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: Tensor of shape [B, query_len, d_model]
            key: Tensor of shape [B, key_len, d_model]
            value: Tensor of shape [B, value_len, d_model]
            mask: Optional additive mask of shape [B, num_heads, query_len, key_len]
                  with 0s for allowed positions and -1e9 for disallowed.
        Returns:
            Tensor of shape [B, query_len, d_model]
        """
        B = query.size(0)
        
        # Linear projections
        q = self.linear_q(query)  # [B, query_len, num_heads*d_k]
        k = self.linear_k(key)      # [B, key_len, num_heads*d_k]
        v = self.linear_v(value)    # [B, value_len, num_heads*d_v]
        
        # Reshape and permute to get [B, num_heads, seq_len, d_k/d_v]
        q = q.view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, self.d_v).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # [B, num_heads, query_len, key_len]
        if mask is not None:
            scores = scores + mask  # mask is additive
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Weighted sum of values
        output = torch.matmul(attn, v)  # [B, num_heads, query_len, d_v]
        
        # Concatenate multiple heads
        output = output.transpose(1, 2).contiguous()  # [B, query_len, num_heads, d_v]
        output = output.view(B, -1, self.num_heads * self.d_v)
        output = self.linear_out(output)
        
        return output


class PositionwiseFeedForward(nn.Module):
    """
    Implements a feed-forward network that is applied pointwise to each position.
    
    Consists of two linear transformations with a ReLU activation in between.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class EncoderLayer(nn.Module):
    """
    One layer of the Transformer encoder.
    
    Comprises a multi-head self-attention sub-layer and a position-wise feed-forward sub-layer,
    each followed by a residual connection and layer normalization.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_k: int,
        d_v: int,
        d_ff: int,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, d_k, d_v, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention sub-layer with residual connection and layer norm
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        # Feed-forward sub-layer with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class Encoder(nn.Module):
    """
    Stack of encoder layers.
    """
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_k: int,
        d_v: int,
        d_ff: int,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_k, d_v, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """
    One layer of the Transformer decoder.
    
    Contains three sub-layers:
      1. Masked multi-head self-attention with combined future and padding mask.
      2. Multi-head encoder–decoder attention.
      3. Position-wise feed-forward network.
      
    Each sub-layer is followed by residual connection and layer normalization.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_k: int,
        d_v: int,
        d_ff: int,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, d_k, d_v, dropout)
        self.enc_dec_attn = MultiHeadAttention(d_model, num_heads, d_k, d_v, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Masked self-attention sub-layer
        x2 = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(x2))
        # Encoder–decoder attention sub-layer
        x2 = self.enc_dec_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(x2))
        # Feed-forward sub-layer
        x2 = self.feed_forward(x)
        x = self.norm3(x + self.dropout(x2))
        return x


class Decoder(nn.Module):
    """
    Stack of decoder layers.
    """
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_k: int,
        d_v: int,
        d_ff: int,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_k, d_v, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask, src_mask)
        return self.norm(x)


class TransformerModel(nn.Module):
    """
    TransformerModel implements the complete encoder-decoder architecture.

    It includes:
      - Source and target embeddings with sinusoidal positional encodings.
      - A stack of encoder and decoder layers.
      - An output projection layer that can share weights with the target embedding.
    
    The configuration parameter controls model hyperparameters and the weight tying option.
    """
    def __init__(self, config: Dict[str, any], src_vocab_size: int, tgt_vocab_size: int) -> None:
        super().__init__()
        # Determine whether to tie embeddings; default is True.
        self.tie_embeddings: bool = config.get("model", {}).get("tie_embeddings", True)
        
        # Select model configuration (default to "base" if available)
        model_config: Dict[str, any] = config.get("model", {}).get("base", {})
        if not model_config:
            model_config = config.get("model", {}).get("big", {})
        
        self.num_layers: int = int(model_config.get("num_layers", 6))
        self.d_model: int = int(model_config.get("d_model", 512))
        self.d_ff: int = int(model_config.get("d_ff", 2048))
        self.num_heads: int = int(model_config.get("num_heads", 8))
        self.d_k: int = int(model_config.get("d_k", 64))
        self.d_v: int = int(model_config.get("d_v", 64))
        self.dropout_rate: float = config.get("training", {}).get("dropout_rate", 0.1)
        
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.scale: float = math.sqrt(self.d_model)
        
        # Embedding layers (with weight tying if enabled and vocab sizes match)
        self.src_embed: nn.Embedding = nn.Embedding(self.src_vocab_size, self.d_model)
        if self.tie_embeddings:
            if self.src_vocab_size != self.tgt_vocab_size:
                import warnings
                warnings.warn("Tying embeddings but source and target vocabulary sizes differ. Using separate embeddings.")
                self.tie_embeddings = False
                self.tgt_embed = nn.Embedding(self.tgt_vocab_size, self.d_model)
            else:
                self.tgt_embed = self.src_embed
        else:
            self.tgt_embed = nn.Embedding(self.tgt_vocab_size, self.d_model)
        
        # Positional encoding module
        self.positional_encoding = PositionalEncoding(self.d_model, self.dropout_rate)
        
        # Encoder and Decoder stacks
        self.encoder = Encoder(
            num_layers=self.num_layers,
            d_model=self.d_model,
            num_heads=self.num_heads,
            d_k=self.d_k,
            d_v=self.d_v,
            d_ff=self.d_ff,
            dropout=self.dropout_rate
        )
        self.decoder = Decoder(
            num_layers=self.num_layers,
            d_model=self.d_model,
            num_heads=self.num_heads,
            d_k=self.d_k,
            d_v=self.d_v,
            d_ff=self.d_ff,
            dropout=self.dropout_rate
        )
        
        # Output projection layer; tie weights if enabled and vocab sizes match
        if self.tie_embeddings and (self.src_vocab_size == self.tgt_vocab_size):
            self.out_proj = nn.Linear(self.d_model, self.tgt_vocab_size)
            self.out_proj.weight = self.tgt_embed.weight
        else:
            self.out_proj = nn.Linear(self.d_model, self.tgt_vocab_size)
    
    def generate_subsequent_mask(self, size: int, device: torch.device) -> torch.Tensor:
        """
        Generates a lower triangular matrix of booleans used to mask out future positions.
        
        Args:
            size: The target sequence length.
            device: The device on which to create the mask.
        Returns:
            A boolean tensor of shape [size, size] where True indicates an allowed position.
        """
        mask = torch.tril(torch.ones((size, size), device=device, dtype=torch.bool))
        return mask

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for the Transformer model.
        
        Args:
            src: Source token tensor with shape [B, src_seq_len].
            tgt: Target token tensor with shape [B, tgt_seq_len].
            src_mask: Optional additive mask for the source (shape [B, 1, 1, src_seq_len]).
                      If None, computed from src tokens (assumes pad token = 0).
            tgt_mask: Not used directly; the decoder mask is computed internally.
        
        Returns:
            Logits tensor of shape [B, tgt_seq_len, tgt_vocab_size].
        """
        device = src.device
        B, src_len = src.size()
        B2, tgt_len = tgt.size()
        
        # Compute source padding mask (assumes pad token is 0)
        if src_mask is None:
            src_padding = (src != 0)  # [B, src_len] bool tensor: True for non-pad tokens
            src_mask = src_padding.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, src_len]
            src_mask = torch.where(src_mask, torch.tensor(0.0, device=device), torch.tensor(-1e9, device=device))
        
        # Compute target padding mask
        tgt_padding = (tgt != 0)  # [B, tgt_len] bool tensor
        
        # Generate subsequent (future) mask for target sequence
        subsequent_mask = self.generate_subsequent_mask(tgt_len, device)  # [tgt_len, tgt_len] bool
        
        # Expand target padding mask to combine with subsequent mask:
        # Expand tgt_padding to [B, 1, tgt_len]
        tgt_padding_exp = tgt_padding.unsqueeze(1)  # [B, 1, tgt_len]
        # The subsequent mask: [1, tgt_len, tgt_len]
        subsequent_mask_exp = subsequent_mask.unsqueeze(0)  # [1, tgt_len, tgt_len]
        # Combine: a position is allowed only if it is not padded and not in the future.
        combined_bool_mask = tgt_padding_exp & subsequent_mask_exp  # [B, tgt_len, tgt_len] bool
        # Unsqueeze to add head dimension: [B, 1, tgt_len, tgt_len]
        combined_bool_mask = combined_bool_mask.unsqueeze(1)
        # Convert boolean mask to additive mask: 0 where allowed, -1e9 where not allowed
        tgt_combined_mask = torch.where(combined_bool_mask, torch.tensor(0.0, device=device), torch.tensor(-1e9, device=device))
        
        # Embedding lookup and scaling
        src_emb = self.src_embed(src) * self.scale  # [B, src_len, d_model]
        tgt_emb = self.tgt_embed(tgt) * self.scale  # [B, tgt_len, d_model]
        # Add positional encodings
        src_emb = self.positional_encoding(src_emb)
        tgt_emb = self.positional_encoding(tgt_emb)
        
        # Encoder forward pass
        encoder_output = self.encoder(src_emb, src_mask)
        # Decoder forward pass (using combined target mask and source mask for encoder-decoder attention)
        decoder_output = self.decoder(tgt_emb, encoder_output, tgt_mask=tgt_combined_mask, src_mask=src_mask)
        
        # Final linear projection to vocabulary space
        logits = self.out_proj(decoder_output)  # [B, tgt_len, tgt_vocab_size]
        return logits
