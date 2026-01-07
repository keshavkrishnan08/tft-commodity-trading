"""
Enhanced LSTM with Variable Selection Network (VSN)

Combines best of both worlds:
- VSN from TFT (automatic feature selection)
- LSTM architecture (temporal modeling)
- Multi-head attention (important timestep identification)

This should achieve positive returns by:
1. VSN filters noise from 219 features
2. LSTM captures temporal patterns
3. Attention focuses on key moments
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.variable_selection import SimpleVariableSelection


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention over LSTM outputs."""
    def __init__(self, hidden_size, num_heads=4, dropout=0.3):
        super(MultiHeadSelfAttention, self).__init__()
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

        return out, attn


class LSTMWithVSN(nn.Module):
    """
    LSTM with Variable Selection Network.

    Architecture:
    1. Per-asset VSN (learns important features for each asset independently)
    2. Bidirectional LSTM (temporal modeling)
    3. Multi-head self-attention (timestep importance)
    4. Asset-specific prediction heads

    Key improvement: VSN reduces 219 features to focused subset,
    eliminating noise and preventing overfitting.
    """
    def __init__(self, config):
        super(LSTMWithVSN, self).__init__()
        self.config = config

        input_dim = config.INPUT_DIM  # 219 features
        hidden_size = config.LSTM_HIDDEN_SIZE  # 128
        num_layers = config.LSTM_LAYERS  # 2
        dropout = config.DROPOUT  # 0.4

        # Per-asset Variable Selection Networks
        # Each asset learns which features are important FOR THAT ASSET
        self.vsn_networks = nn.ModuleDict({
            asset: SimpleVariableSelection(
                num_features=input_dim,
                hidden_dim=hidden_size,
                output_dim=hidden_size,
                dropout=dropout
            )
            for asset in config.TARGET_ASSETS
        })

        # Shared bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,  # VSN output dimension
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )

        # Project bidirectional to single direction
        self.lstm_proj = nn.Linear(hidden_size * 2, hidden_size)

        # Multi-head self-attention
        self.self_attention = MultiHeadSelfAttention(
            hidden_size=hidden_size,
            num_heads=4,
            dropout=dropout
        )

        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Per-asset prediction heads
        self.classifiers = nn.ModuleDict({
            asset: nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, hidden_size // 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 4, 1)
            )
            for asset in config.TARGET_ASSETS
        })

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim=219)

        Returns:
            predictions: dict of {asset: logits (batch, 1)}
            feature_importances: dict of {asset: (batch, seq_len, input_dim)}
        """
        batch_size, seq_len, input_dim = x.shape
        predictions = {}
        feature_importances = {}

        for asset in self.config.TARGET_ASSETS:
            # 1. Variable Selection (most important step!)
            # Reduces 219 features to focused hidden_dim representation
            selected, weights = self.vsn_networks[asset](x)  # (batch, seq_len, hidden_size)
            feature_importances[asset] = weights

            # 2. LSTM encoding
            lstm_out, _ = self.lstm(selected)  # (batch, seq_len, hidden_size*2)
            lstm_out = self.lstm_proj(lstm_out)  # (batch, seq_len, hidden_size)
            lstm_out = self.layer_norm1(lstm_out)
            lstm_out = self.dropout(lstm_out)

            # 3. Self-attention
            attn_out, _ = self.self_attention(lstm_out)

            # Residual connection
            encoded = self.layer_norm2(lstm_out + attn_out)

            # 4. Global pooling
            pooled = encoded.mean(dim=1)  # (batch, hidden_size)

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
    config.DROPOUT = 0.4

    model = LSTMWithVSN(config)

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
    print("✓ LSTM with VSN Test Passed!")
