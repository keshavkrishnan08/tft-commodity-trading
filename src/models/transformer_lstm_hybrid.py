"""
Transformer-LSTM Hybrid with VSN

Replaces the failed TCN architecture with a proven combination:
1. Variable Selection Network (noise reduction)
2. Multi-head Self-Attention (like Transformers - capture long-range dependencies)
3. LSTM layers (temporal modeling)
4. Residual connections
5. Layer normalization

This should significantly outperform TCN by combining the best aspects of
modern architectures.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.variable_selection import SimpleVariableSelection


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    def __init__(self, hidden_size, num_heads=4, dropout=0.3):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads

        assert hidden_size % num_heads == 0

        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # QKV projections
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention
        out = attn @ v
        out = out.transpose(1, 2).contiguous().reshape(batch_size, seq_len, self.hidden_size)
        out = self.out_proj(out)

        return out


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    def __init__(self, hidden_size, ff_dim, dropout=0.3):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(hidden_size, ff_dim)
        self.fc2 = nn.Linear(ff_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Single Transformer block with attention + FFN."""
    def __init__(self, hidden_size, num_heads=4, ff_dim=256, dropout=0.3):
        super(TransformerBlock, self).__init__()

        self.attention = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ff = FeedForward(hidden_size, ff_dim, dropout)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Multi-head attention with residual
        attn_out = self.attention(x)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward with residual
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))

        return x


class TransformerLSTMHybrid(nn.Module):
    """
    Hybrid architecture combining Transformers and LSTMs with VSN.

    Architecture:
    1. Per-asset VSN (feature selection)
    2. Transformer blocks (capture long-range dependencies)
    3. Bidirectional LSTM (temporal modeling)
    4. Asset-specific prediction heads

    This should dramatically outperform TCN by using proven architectures.
    """
    def __init__(self, config):
        super(TransformerLSTMHybrid, self).__init__()
        self.config = config

        input_dim = config.INPUT_DIM  # 219
        hidden_size = config.LSTM_HIDDEN_SIZE  # 128
        num_layers = config.LSTM_LAYERS  # 2
        dropout = config.DROPOUT  # 0.3

        # Per-asset Variable Selection Networks
        self.vsn_networks = nn.ModuleDict({
            asset: SimpleVariableSelection(
                num_features=input_dim,
                hidden_dim=hidden_size,
                output_dim=hidden_size,
                dropout=dropout
            )
            for asset in config.TARGET_ASSETS
        })

        # Transformer blocks (2 blocks for long-range dependencies)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size=hidden_size,
                num_heads=4,
                ff_dim=hidden_size * 2,
                dropout=dropout
            )
            for _ in range(2)  # 2 transformer blocks
        ])

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )

        # Project bidirectional output back to hidden_size
        self.lstm_proj = nn.Linear(hidden_size * 2, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Per-asset prediction heads
        self.classifiers = nn.ModuleDict({
            asset: nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.LayerNorm(hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, 1)
            )
            for asset in config.TARGET_ASSETS
        })

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim=219)

        Returns:
            predictions: dict of {asset: logits (batch, 1)}
        """
        batch_size, seq_len, input_dim = x.shape
        predictions = {}

        for asset in self.config.TARGET_ASSETS:
            # 1. Variable Selection (noise reduction)
            selected, weights = self.vsn_networks[asset](x)  # (batch, seq_len, hidden_size)

            # 2. Transformer blocks (long-range dependencies)
            h = selected
            for transformer_block in self.transformer_blocks:
                h = transformer_block(h)  # (batch, seq_len, hidden_size)

            # 3. LSTM (temporal modeling)
            lstm_out, _ = self.lstm(h)  # (batch, seq_len, hidden_size*2)
            lstm_out = self.lstm_proj(lstm_out)  # (batch, seq_len, hidden_size)
            lstm_out = self.layer_norm(lstm_out)

            # 4. Global pooling (combine all timesteps)
            # Use both max and mean pooling for richer representation
            max_pool = lstm_out.max(dim=1)[0]  # (batch, hidden_size)
            mean_pool = lstm_out.mean(dim=1)  # (batch, hidden_size)
            pooled = (max_pool + mean_pool) / 2  # Average of both

            # 5. Prediction
            pred = self.classifiers[asset](pooled)
            predictions[asset] = pred

        return predictions


if __name__ == "__main__":
    # Test the model
    from src.utils.config import Config

    config = Config()
    config.INPUT_DIM = 219
    config.LSTM_HIDDEN_SIZE = 128
    config.LSTM_LAYERS = 2
    config.DROPOUT = 0.3

    model = TransformerLSTMHybrid(config)

    # Test forward pass
    batch_size = 32
    seq_len = 60
    x = torch.randn(batch_size, seq_len, config.INPUT_DIM)

    outputs = model(x)

    print(f"Input shape: {x.shape}")
    for asset, out in outputs.items():
        print(f"{asset} output shape: {out.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print("✓ Transformer-LSTM Hybrid Test Passed!")
