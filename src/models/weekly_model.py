# src/models/weekly_model.py
"""
Weekly/Monthly Prediction Model with Full Sequence Analysis.

Key improvements:
1. Uses FULL sequence (all timesteps), not just last one
2. Temporal attention to find important past periods
3. Cross-asset attention to find relationships
4. Designed for longer-horizon predictions
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import Config


class TemporalAttention(nn.Module):
    """
    Attention over TIME to find important historical patterns.
    """
    def __init__(self, hidden_size, num_heads=4, dropout=0.3):
        super(TemporalAttention, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, hidden]

        Returns:
            output: [batch, seq_len, hidden]
            weights: [batch, seq_len, seq_len]
        """
        attn_output, attn_weights = self.attention(x, x, x)
        output = self.layer_norm(x + self.dropout(attn_output))
        return output, attn_weights


class CrossAssetAttention(nn.Module):
    """
    Attention across ASSETS to learn relationships.
    Each asset attends to all other assets.
    """
    def __init__(self, hidden_size, num_heads=4, dropout=0.3):
        super(CrossAssetAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        """
        Args:
            x: [batch, num_assets, hidden]

        Returns:
            output: [batch, num_assets, hidden]
            attention_weights: [batch, num_assets, num_assets]
        """
        batch_size, num_assets, hidden_size = x.shape

        Q = self.q_proj(x).view(batch_size, num_assets, self.num_heads, self.head_dim)
        K = self.k_proj(x).view(batch_size, num_assets, self.num_heads, self.head_dim)
        V = self.v_proj(x).view(batch_size, num_assets, self.num_heads, self.head_dim)

        Q = Q.transpose(1, 2)  # [batch, heads, num_assets, head_dim]
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(batch_size, num_assets, hidden_size)

        output = self.out_proj(attended)
        output = self.layer_norm(x + self.dropout(output))

        # Average across heads for interpretability
        attention_weights_avg = attention_weights.mean(dim=1)

        return output, attention_weights_avg


class AssetLSTM(nn.Module):
    """LSTM encoder for a single asset's time series"""
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(AssetLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, input_size]

        Returns:
            output: [batch, seq_len, hidden_size]
        """
        output, (h_n, c_n) = self.lstm(x)
        return output


class WeeklyPredictionModel(nn.Module):
    """
    Model for weekly/monthly directional predictions.

    Architecture:
        1. Per-asset LSTM to encode time series
        2. Temporal attention over sequence (find important periods)
        3. Cross-asset attention (learn asset relationships)
        4. Per-asset classification heads
    """
    def __init__(self, config=None):
        super(WeeklyPredictionModel, self).__init__()

        self.config = config if config else Config()

        # Calculate features per asset based on new indicators
        # Base: Return, LogReturn
        # Volatility: 4 windows (5, 10, 20, 60)
        # SMA: 4 windows (10, 20, 50, 200)
        # EMA: 3 windows (12, 26, 50)
        # RSI: 1
        # MACD: 3 (MACD, Signal, Hist)
        # Divergences: 4 SMA + 3 EMA = 7
        # Momentum: 2 (20d, 60d)
        # Total: 2 + 4 + 4 + 3 + 1 + 3 + 7 + 2 = 26 features per asset
        self.features_per_asset = 26

        # Moderate dropout to avoid overfitting
        dropout = 0.3

        # Per-asset LSTM encoders
        self.asset_lstms = nn.ModuleList([
            AssetLSTM(
                input_size=self.features_per_asset,
                hidden_size=self.config.LSTM_HIDDEN_SIZE,
                num_layers=self.config.LSTM_LAYERS,
                dropout=dropout
            )
            for _ in self.config.ALL_ASSETS
        ])

        # Temporal attention (over time)
        self.temporal_attention = TemporalAttention(
            hidden_size=self.config.LSTM_HIDDEN_SIZE,
            num_heads=4,
            dropout=dropout
        )

        # Cross-asset attention
        self.cross_asset_attention = CrossAssetAttention(
            hidden_size=self.config.LSTM_HIDDEN_SIZE,
            num_heads=4,
            dropout=dropout
        )

        # Classification heads
        # NOTE: Output raw logits (no sigmoid) - loss function applies sigmoid via BCEWithLogitsLoss
        self.classifiers = nn.ModuleDict({
            asset: nn.Sequential(
                nn.Linear(self.config.LSTM_HIDDEN_SIZE, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1)
                # No Sigmoid here - BCEWithLogitsLoss applies it internally for numerical stability
            )
            for asset in self.config.TARGET_ASSETS
        })

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, num_assets * features_per_asset]

        Returns:
            predictions: dict of {asset: probability [batch, 1]}
            attention_weights: [batch, num_assets, num_assets]
        """
        batch_size, seq_len, total_features = x.shape

        # Process each asset through its LSTM
        asset_sequences = []
        for i, asset in enumerate(self.config.ALL_ASSETS):
            start_idx = i * self.features_per_asset
            end_idx = start_idx + self.features_per_asset
            asset_features = x[:, :, start_idx:end_idx]

            # LSTM encoding: [batch, seq_len, hidden]
            lstm_out = self.asset_lstms[i](asset_features)
            asset_sequences.append(lstm_out)

        # Stack: [batch, num_assets, seq_len, hidden]
        stacked = torch.stack(asset_sequences, dim=1)

        # Apply temporal attention to each asset independently
        temporal_outputs = []
        for i in range(len(self.config.ALL_ASSETS)):
            asset_seq = stacked[:, i, :, :]  # [batch, seq_len, hidden]
            attended_seq, _ = self.temporal_attention(asset_seq)
            # Global average pool over time
            pooled = attended_seq.mean(dim=1)  # [batch, hidden]
            temporal_outputs.append(pooled)

        # Stack asset representations: [batch, num_assets, hidden]
        asset_reprs = torch.stack(temporal_outputs, dim=1)

        # Cross-asset attention
        attended_assets, attention_weights = self.cross_asset_attention(asset_reprs)

        # Predict for each target asset
        predictions = {}
        for i, asset in enumerate(self.config.TARGET_ASSETS):
            asset_repr = attended_assets[:, i, :]  # [batch, hidden]
            pred = self.classifiers[asset](asset_repr)
            predictions[asset] = pred

        return predictions, attention_weights

    def get_attention_weights_per_asset(self, attention_weights):
        """Already at asset level"""
        return attention_weights


if __name__ == "__main__":
    # Test model
    config = Config()
    model = WeeklyPredictionModel(config)

    # Create dummy input
    batch_size = 32
    seq_len = config.SEQUENCE_LENGTH
    num_assets = len(config.ALL_ASSETS)
    features_per_asset = 26  # Updated feature count
    total_features = num_assets * features_per_asset

    x = torch.randn(batch_size, seq_len, total_features)

    # Forward pass
    predictions, attention_weights = model(x)

    print("="*80)
    print("WEEKLY PREDICTION MODEL TEST")
    print("="*80)
    print(f"Input shape: {x.shape}")
    print(f"Features per asset: {features_per_asset}")
    print(f"\nPredictions:")
    for asset, pred in predictions.items():
        print(f"  {asset}: {pred.shape}")
    print(f"\nAttention weights shape: {attention_weights.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print("="*80)
