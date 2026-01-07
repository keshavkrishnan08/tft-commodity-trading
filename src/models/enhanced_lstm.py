"""
Enhanced LSTM with Global Feature Attention

Uses same 199 calibrated features as TFT but with LSTM architecture.
Key improvements over SimpleLSTM:
- Bidirectional LSTM for better temporal modeling
- Multi-head self-attention over LSTM outputs
- Separate feature importance learning per asset
- Layer normalization for training stability
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention over LSTM sequence outputs.
    Helps identify important timesteps for prediction.
    """
    def __init__(self, hidden_size, num_heads=4, dropout=0.3):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads

        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, hidden_size)
        Returns:
            out: (batch, seq_len, hidden_size)
            attn_weights: (batch, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape

        # QKV projections
        qkv = self.qkv(x)  # (batch, seq_len, 3*hidden_size)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (batch, num_heads, seq_len, seq_len)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention
        out = attn @ v  # (batch, num_heads, seq_len, head_dim)
        out = out.transpose(1, 2).contiguous()  # (batch, seq_len, num_heads, head_dim)
        out = out.reshape(batch_size, seq_len, self.hidden_size)
        out = self.out_proj(out)

        return out, attn


class FeatureImportanceLayer(nn.Module):
    """
    Learn feature importance weights for each asset.
    Different assets may care about different features.
    """
    def __init__(self, input_dim, hidden_dim):
        super(FeatureImportanceLayer, self).__init__()
        self.importance_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # Outputs weights [0, 1]
        )

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            weighted_x: (batch, seq_len, input_dim)
            importance: (batch, input_dim) - average importance across sequence
        """
        # Global average pool over time to get feature statistics
        pooled = x.mean(dim=1)  # (batch, input_dim)

        # Learn importance weights
        importance = self.importance_net(pooled)  # (batch, input_dim)

        # Apply weights
        weighted_x = x * importance.unsqueeze(1)  # (batch, seq_len, input_dim)

        return weighted_x, importance


class EnhancedLSTM(nn.Module):
    """
    Enhanced LSTM for multi-asset directional prediction.

    Architecture:
    1. Per-asset feature importance learning
    2. Bidirectional LSTM encoder
    3. Multi-head self-attention over LSTM outputs
    4. Per-asset prediction heads with residual connection

    Same feature input as TFT (199 calibrated features) but LSTM-based architecture.
    """
    def __init__(self, config):
        super(EnhancedLSTM, self).__init__()
        self.config = config

        input_dim = config.INPUT_DIM
        hidden_size = config.LSTM_HIDDEN_SIZE
        num_layers = config.LSTM_LAYERS
        dropout = config.DROPOUT

        # Per-asset feature importance learning
        self.feature_importance = nn.ModuleDict({
            asset: FeatureImportanceLayer(input_dim, hidden_size)
            for asset in config.TARGET_ASSETS
        })

        # Shared bidirectional LSTM encoder
        # Bidirectional: better temporal modeling (future AND past context)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )

        # Project bidirectional outputs to hidden_size
        self.lstm_proj = nn.Linear(hidden_size * 2, hidden_size)

        # Multi-head self-attention
        self.self_attention = MultiHeadSelfAttention(
            hidden_size=hidden_size,
            num_heads=4,
            dropout=dropout
        )

        # Layer normalization for training stability
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)

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
            x: (batch, seq_len, input_dim)

        Returns:
            predictions: dict of {asset: logits (batch, 1)}
        """
        batch_size = x.size(0)
        predictions = {}

        # Process each asset
        for asset in self.config.TARGET_ASSETS:
            # 1. Feature importance learning
            weighted_x, importance = self.feature_importance[asset](x)

            # 2. LSTM encoding
            lstm_out, (h_n, c_n) = self.lstm(weighted_x)  # (batch, seq_len, hidden_size*2)

            # Project to hidden_size
            lstm_out = self.lstm_proj(lstm_out)  # (batch, seq_len, hidden_size)
            lstm_out = self.layer_norm1(lstm_out)

            # 3. Self-attention over LSTM outputs
            attn_out, attn_weights = self.self_attention(lstm_out)

            # Residual connection
            encoded = self.layer_norm2(lstm_out + attn_out)

            # 4. Global pooling over time
            pooled = encoded.mean(dim=1)  # (batch, hidden_size)

            # 5. Prediction
            pred = self.classifiers[asset](pooled)
            predictions[asset] = pred

        return predictions


if __name__ == "__main__":
    # Test the model
    from src.utils.config import Config

    config = Config()
    config.INPUT_DIM = 199
    config.LSTM_HIDDEN_SIZE = 128
    config.LSTM_LAYERS = 2
    config.DROPOUT = 0.4

    model = EnhancedLSTM(config)

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
    print("✓ EnhancedLSTM Test Passed!")
