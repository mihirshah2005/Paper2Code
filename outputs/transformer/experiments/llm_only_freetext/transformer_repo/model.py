"""model.py

This module implements the full Transformer architecture as described in "Attention Is All You Need."
It includes all helper modules relevant to the encoder–decoder Transformer network:
• PositionalEncoding
• MultiHeadAttention
• PositionwiseFeedForward
• EncoderLayer and Encoder
• DecoderLayer and Decoder
• TransformerModel (which ties together the encoder and decoder and uses a shared embedding layer)

All hyperparameters (e.g., d_model, d_ff, num_layers, num_heads, d_k, d_v, dropout) are read from a configuration
dictionary (derived from "config.yaml"). Default values are provided as fallback if a setting is missing.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Utility Masking Functions
# ---------------------------
def generate_subsequent_mask(sz: int) -> torch.Tensor:
    """
    Generates a subsequent mask for a sequence of length sz.
    This mask prohibits each position from attending to subsequent positions.
    Returns:
        mask: Tensor of shape (sz, sz), where positions (i, j) are set to 0 if j <= i and -inf otherwise.
    """
    mask = torch.triu(torch.ones(sz, sz), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask


def create_padding_mask(seq: torch.Tensor, pad_token: int = 0) -> torch.Tensor:
    """
    Creates a mask for padded positions in the sequence.
    Args:
        seq: Tensor of shape (batch_size, seq_len)
        pad_token: The integer representing the pad token.
    Returns:
        mask: Tensor of shape (batch_size, 1, 1, seq_len) (broadcastable for attention scores)
    """
    # Mask is True (or 1) for positions equal to pad_token. We convert this mask to float and set  -inf where padded.
    mask = (seq == pad_token)
    return mask.unsqueeze(1).unsqueeze(2)  # broadcastable mask


# ---------------------------
# Positional Encoding Module
# ---------------------------
class PositionalEncoding(nn.Module):
    """
    Implements sinusoidal positional encoding.
    Given a fixed max_len and d_model, this module precomputes a positional encoding matrix
    which is added to the input embeddings.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        """
        Args:
            d_model: Dimension of the embeddings.
            dropout: Dropout probability to be applied after adding positional encoding.
            max_len: Maximum length of sequences for which to precompute the positional encoding.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Create a long enough 'pe' matrix with shape [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Add an additional batch dimension for easier addition.
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input embeddings of shape (batch_size, seq_len, d_model)
        Returns:
            Output tensor with positional encodings added (same shape as x)
        """
        seq_len = x.size(1)
        # Add precomputed positional encodings to input; note pe already has shape (1, max_len, d_model)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


# ---------------------------
# Multi-Head Attention Module
# ---------------------------
class MultiHeadAttention(nn.Module):
    """
    Implements multi-head scaled dot-product attention.
    Projects queries, keys, and values, splits them into multiple heads,
    applies the scaled dot-product attention on each head separately, and concatenates the results.
    """
    def __init__(self, d_model: int, num_heads: int, d_k: int, d_v: int, dropout: float = 0.1) -> None:
        """
        Args:
            d_model: Total dimension of the model.
            num_heads: Number of attention heads.
            d_k: Dimension of keys (per head).
            d_v: Dimension of values (per head).
            dropout: Dropout rate applied to the attention weights.
        """
        super(MultiHeadAttention, self).__init__()
        if d_model != num_heads * d_k:
            raise ValueError("d_model must be equal to num_heads * d_k")
        if d_model != num_heads * d_v:
            raise ValueError("d_model must be equal to num_heads * d_v")
            
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        
        self.linear_q = nn.Linear(d_model, num_heads * d_k)
        self.linear_k = nn.Linear(d_model, num_heads * d_k)
        self.linear_v = nn.Linear(d_model, num_heads * d_v)
        self.linear_out = nn.Linear(num_heads * d_v, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            query: Tensor of shape (batch_size, seq_len_q, d_model)
            key: Tensor of shape (batch_size, seq_len_k, d_model)
            value: Tensor of shape (batch_size, seq_len_v, d_model)
            mask: Optional mask tensor that is broadcastable to shape (batch_size, num_heads, seq_len_q, seq_len_k)
        Returns:
            Output tensor of shape (batch_size, seq_len_q, d_model)
        """
        batch_size = query.size(0)

        # Apply linear projections and reshape for multiple heads
        # New shape: (batch_size, num_heads, seq_len, d_k or d_v)
        Q = self.linear_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.linear_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.linear_v(value).view(batch_size, -1, self.num_heads, self.d_v).transpose(1, 2)

        # Compute scaled dot-product attention scores.
        # Q shape: (batch_size, num_heads, seq_len_q, d_k)
        # K shape: (batch_size, num_heads, seq_len_k, d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # shape: (batch_size, num_heads, seq_len_q, seq_len_k)
        
        # Apply mask if provided
        if mask is not None:
            # mask should be broadcastable to scores' shape
            scores = scores + mask

        # Softmax over the last dimension (seq_len_k)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Compute attention output
        attn_output = torch.matmul(attn_weights, V)  # shape: (batch_size, num_heads, seq_len_q, d_v)
        # Concatenate all heads
        attn_output = attn_output.transpose(1, 2).contiguous()  # shape: (batch_size, seq_len_q, num_heads, d_v)
        attn_output = attn_output.view(batch_size, -1, self.num_heads * self.d_v)  # shape: (batch_size, seq_len_q, d_model)
        # Final linear projection
        output = self.linear_out(attn_output)
        return output


# ---------------------------
# Position-wise Feed-Forward Module
# ---------------------------
class PositionwiseFeedForward(nn.Module):
    """
    Implements the position-wise feed-forward network.
    Consists of two linear transformations with a ReLU activation in between.
    Applies the same feed-forward network independently to each position.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        """
        Args:
            d_model: Input and output dimension.
            d_ff: Inner layer dimension.
            dropout: Dropout rate used after the first linear transformation.
        """
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor of the same shape after applying feed-forward transformations.
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ---------------------------
# Encoder Layer Module
# ---------------------------
class EncoderLayer(nn.Module):
    """
    Implements one layer of the Transformer encoder.
    Consists of a multi-head self-attention sub-layer and a position-wise feed-forward sub-layer,
    each followed by a residual connection and layer normalization.
    """
    def __init__(self, d_model: int, num_heads: int, d_k: int, d_v: int, d_ff: int, dropout: float = 0.1) -> None:
        """
        Args:
            d_model: Model dimensionality.
            num_heads: Number of attention heads.
            d_k: Dimension of keys.
            d_v: Dimension of values.
            d_ff: Dimension of the feed-forward network (inner layer).
            dropout: Dropout rate.
        """
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, d_k=d_k, d_v=d_v, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask tensor for self-attention.
        Returns:
            Tensor of the same shape after processing.
        """
        # Self-Attention sub-layer with residual connection and layer normalization.
        attn_output = self.self_attn(x, x, x, mask=mask)
        x = self.norm1(x + self.dropout(attn_output))
        # Feed-Forward sub-layer with residual connection and layer normalization.
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


# ---------------------------
# Encoder Module
# ---------------------------
class Encoder(nn.Module):
    """
    Implements the Transformer encoder.
    It embeds the input tokens, adds positional encodings,
    and passes the results through a stack of EncoderLayer modules.
    """
    def __init__(self, num_layers: int, vocab_size: int, d_model: int, d_ff: int,
                 num_heads: int, d_k: int, d_v: int, dropout: float = 0.1,
                 max_len: int = 5000, shared_embedding: Optional[nn.Embedding] = None) -> None:
        """
        Args:
            num_layers: Number of encoder layers.
            vocab_size: Size of the input vocabulary.
            d_model: Model dimensionality.
            d_ff: Internal dimensionality of the feed-forward network.
            num_heads: Number of attention heads.
            d_k: Dimension of attention keys.
            d_v: Dimension of attention values.
            dropout: Dropout rate.
            max_len: Maximum sequence length for positional encoding.
            shared_embedding: (Optional) Shared embedding layer. If provided, it will be used instead of creating a new one.
        """
        super(Encoder, self).__init__()
        self.d_model = d_model
        if shared_embedding is None:
            self.embedding = nn.Embedding(vocab_size, d_model)
        else:
            self.embedding = shared_embedding
        
        self.positional_encoding = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=max_len)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model=d_model, num_heads=num_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            src: Source token ids tensor of shape (batch_size, seq_len)
            mask: Optional mask tensor for padding positions.
        Returns:
            Encoder output tensor of shape (batch_size, seq_len, d_model)
        """
        # Embed tokens and scale by sqrt(d_model)
        embedded = self.embedding(src) * math.sqrt(self.d_model)
        # Add positional encodings
        x = self.positional_encoding(embedded)
        # Pass through stacked encoder layers
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# ---------------------------
# Decoder Layer Module
# ---------------------------
class DecoderLayer(nn.Module):
    """
    Implements one layer of the Transformer decoder.
    Consists of:
     - Masked multi-head self-attention sub-layer.
     - Multi-head attention sub-layer over encoder outputs.
     - Position-wise feed-forward sub-layer.
    Each sub-layer is followed by residual connections and layer normalization.
    """
    def __init__(self, d_model: int, num_heads: int, d_k: int, d_v: int, d_ff: int, dropout: float = 0.1) -> None:
        """
        Args:
            d_model: Model dimensionality.
            num_heads: Number of attention heads.
            d_k: Dimension of keys.
            d_v: Dimension of values.
            d_ff: Internal dimension for the feed-forward network.
            dropout: Dropout rate.
        """
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, d_k=d_k, d_v=d_v, dropout=dropout)
        self.enc_dec_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, d_k=d_k, d_v=d_v, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None, memory_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Target input tensor of shape (batch_size, tgt_seq_len, d_model)
            memory: Encoder output tensor of shape (batch_size, src_seq_len, d_model)
            tgt_mask: Optional mask for target sequence (e.g. subsequent mask).
            memory_mask: Optional mask for encoder output.
        Returns:
            Tensor of shape (batch_size, tgt_seq_len, d_model) after processing through the layer.
        """
        # Masked self-attention sub-layer
        self_attn_output = self.self_attn(x, x, x, mask=tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        # Encoder-decoder attention sub-layer
        enc_dec_attn_output = self.enc_dec_attn(x, memory, memory, mask=memory_mask)
        x = self.norm2(x + self.dropout(enc_dec_attn_output))
        # Feed-forward sub-layer
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


# ---------------------------
# Decoder Module
# ---------------------------
class Decoder(nn.Module):
    """
    Implements the Transformer decoder.
    Embeds the target tokens using the (optionally shared) embedding layer,
    adds positional encodings, and passes the results through a stack of DecoderLayer modules.
    """
    def __init__(self, num_layers: int, vocab_size: int, d_model: int, d_ff: int,
                 num_heads: int, d_k: int, d_v: int, dropout: float = 0.1,
                 max_len: int = 5000, shared_embedding: Optional[nn.Embedding] = None) -> None:
        """
        Args:
            num_layers: Number of decoder layers.
            vocab_size: Vocabulary size for target tokens.
            d_model: Model dimensionality.
            d_ff: Inner dimension of the feed-forward network.
            num_heads: Number of attention heads.
            d_k: Dimension of attention keys.
            d_v: Dimension of attention values.
            dropout: Dropout rate.
            max_len: Maximum sequence length for positional encoding.
            shared_embedding: (Optional) Shared embedding layer; if provided, used for both encoder and decoder.
        """
        super(Decoder, self).__init__()
        self.d_model = d_model
        if shared_embedding is None:
            self.embedding = nn.Embedding(vocab_size, d_model)
        else:
            self.embedding = shared_embedding
        self.positional_encoding = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=max_len)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model=d_model, num_heads=num_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None, memory_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            tgt: Target token ids tensor of shape (batch_size, tgt_seq_len)
            memory: Encoder output tensor of shape (batch_size, src_seq_len, d_model)
            tgt_mask: Optional mask for target tokens (e.g. subsequent mask).
            memory_mask: Optional mask for encoder-decoder attention.
        Returns:
            Tensor of shape (batch_size, tgt_seq_len, d_model) after processing through the decoder stack.
        """
        # Embed target tokens and scale
        embedded = self.embedding(tgt) * math.sqrt(self.d_model)
        x = self.positional_encoding(embedded)
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        return self.norm(x)


# ---------------------------
# Transformer Model
# ---------------------------
class TransformerModel(nn.Module):
    """
    Implements the complete Transformer model as an encoder-decoder architecture.
    It combines a shared embedding layer, the encoder, the decoder, and a final linear projection.
    The output projection layer shares its weights with the input embedding layer (weight tying).
    """
    def __init__(self, config: dict, model_type: str = "base", vocab_size: int = 37000, max_len: int = 5000) -> None:
        """
        Args:
            config: Configuration dictionary (parsed from config.yaml).
            model_type: Either "base" or "big" to select model hyperparameters.
            vocab_size: Vocabulary size used for both encoder and decoder.
            max_len: Maximum sequence length for positional encoding.
        """
        super(TransformerModel, self).__init__()
        
        # Select model configuration based on model_type; defaults to "base"
        model_config = config.get("model", {}).get(model_type, {})
        self.num_layers: int = int(model_config.get("num_layers", 6))
        self.d_model: int = int(model_config.get("d_model", 512))
        self.d_ff: int = int(model_config.get("d_ff", 2048))
        self.num_heads: int = int(model_config.get("num_heads", 8))
        self.d_k: int = int(model_config.get("d_k", 64))
        self.d_v: int = int(model_config.get("d_v", 64))
        self.dropout: float = float(config.get("training", {}).get("dropout_rate", 0.1))
        
        # Create shared embedding layer (for tying input and output embeddings)
        self.shared_embedding = nn.Embedding(vocab_size, self.d_model)
        
        # Build Encoder and Decoder using the shared embedding layer.
        self.encoder = Encoder(
            num_layers=self.num_layers,
            vocab_size=vocab_size,
            d_model=self.d_model,
            d_ff=self.d_ff,
            num_heads=self.num_heads,
            d_k=self.d_k,
            d_v=self.d_v,
            dropout=self.dropout,
            max_len=max_len,
            shared_embedding=self.shared_embedding
        )
        
        self.decoder = Decoder(
            num_layers=self.num_layers,
            vocab_size=vocab_size,
            d_model=self.d_model,
            d_ff=self.d_ff,
            num_heads=self.num_heads,
            d_k=self.d_k,
            d_v=self.d_v,
            dropout=self.dropout,
            max_len=max_len,
            shared_embedding=self.shared_embedding
        )
        
        # Final output projection layer: projects decoder outputs to vocabulary dimensions.
        self.output_projection = nn.Linear(self.d_model, vocab_size)
        
        # Tie weights: share output_projection weights with shared_embedding weights.
        self.output_projection.weight = self.shared_embedding.weight

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None, tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            src: Source token ids tensor of shape (batch_size, src_seq_len)
            tgt: Target token ids tensor of shape (batch_size, tgt_seq_len)
            src_mask: Optional mask for source tokens (e.g., padding mask).
            tgt_mask: Optional mask for target tokens (e.g., subsequent mask and padding mask).
        Returns:
            Logits tensor of shape (batch_size, tgt_seq_len, vocab_size)
        """
        # Encode the source sequence.
        memory = self.encoder(src, mask=src_mask)
        
        # Decode the target sequence; Note: tgt_mask should combine subsequent mask with any padding mask.
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=src_mask)
        
        # Project decoder outputs to vocabulary dimension.
        logits = self.output_projection(output)
        return logits
