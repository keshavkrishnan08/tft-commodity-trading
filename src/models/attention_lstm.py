# src/models/attention_lstm.py
"""
Multi-Asset Directional Prediction Model with Attention
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import Config

class MultiAssetAttention(nn.Module):
    """
    Multi-head attention mechanism across assets and time.
    Learns leading-lag relationships and divergence patterns.
    """
    def __init__(self, hidden_size, num_heads=8, dropout=0.3):
        super(MultiAssetAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.head_dim = hidden_size // num_heads

        # Multi-head attention
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
            x: [batch, seq_len, hidden_size]

        Returns:
            attended: [batch, seq_len, hidden_size]
            weights: [batch, num_heads, seq_len, seq_len]
        """
        # Multi-head self-attention
        attn_output, attn_weights = self.attention(x, x, x)

        # Residual + LayerNorm
        x = self.layer_norm(x + self.dropout(attn_output))

        return x, attn_weights


class AssetLSTM(nn.Module):
    """
    LSTM encoder for a single asset's time series.
    """
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
            (h_n, c_n): final hidden states
        """
        output, (h_n, c_n) = self.lstm(x)
        return output, (h_n, c_n)


class MultiAssetDirectionalModel(nn.Module):
    """
    Multi-Asset Directional Prediction Model with LSTM + Attention.

    Architecture:
        1. Separate LSTM for each asset
        2. Multi-head attention across assets
        3. Binary classification head for each target asset
        4. Attention weights for interpretability
    """
    def __init__(self, config=None, features_per_asset=None):
        super(MultiAssetDirectionalModel, self).__init__()

        self.config = config if config else Config()

        # Number of features per asset - dynamically computed
        # Default to 24 based on current feature engineering (was incorrectly hardcoded as 14)
        self.features_per_asset = features_per_asset if features_per_asset else 24

        # LSTM encoders for each asset
        self.asset_lstms = nn.ModuleList([
            AssetLSTM(
                input_size=self.features_per_asset,
                hidden_size=self.config.LSTM_HIDDEN_SIZE,
                num_layers=self.config.LSTM_LAYERS,
                dropout=self.config.DROPOUT
            )
            for _ in self.config.ALL_ASSETS
        ])

        # Multi-asset attention
        self.attention = MultiAssetAttention(
            hidden_size=self.config.LSTM_HIDDEN_SIZE,
            num_heads=self.config.ATTENTION_HEADS,
            dropout=self.config.DROPOUT
        )

        # Classification heads (one per target asset)
        # NOTE: Output raw logits (no sigmoid) - loss function will apply sigmoid
        self.classifiers = nn.ModuleDict({
            asset: nn.Sequential(
                nn.Linear(self.config.LSTM_HIDDEN_SIZE, 64),
                nn.ReLU(),
                nn.Dropout(self.config.DROPOUT),
                nn.Linear(64, 1)
                # No Sigmoid here - BCEWithLogitsLoss applies it internally
            )
            for asset in self.config.TARGET_ASSETS
        })

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, num_assets * features_per_asset]

        Returns:
            predictions: dict of {asset: probability [batch, 1]}
            attention_weights: [batch, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, total_features = x.shape

        # Process each asset through its own LSTM
        asset_outputs = []
        for i, asset in enumerate(self.config.ALL_ASSETS):
            # Extract features for this asset
            start_idx = i * self.features_per_asset
            end_idx = start_idx + self.features_per_asset
            asset_features = x[:, :, start_idx:end_idx]  # [batch, seq_len, features_per_asset]

            # LSTM encoding
            lstm_out, _ = self.asset_lstms[i](asset_features)  # [batch, seq_len, hidden]
            asset_outputs.append(lstm_out)

        # Stack asset outputs: [batch, seq_len, num_assets, hidden]
        stacked_outputs = torch.stack(asset_outputs, dim=2)

        # Reshape for attention: [batch, seq_len*num_assets, hidden]
        # This allows attention to learn cross-asset and temporal dependencies
        reshaped = stacked_outputs.reshape(batch_size, seq_len * len(self.config.ALL_ASSETS), -1)

        # Multi-asset attention
        attended, attention_weights = self.attention(reshaped)

        # Global pooling across time and assets
        # Average pool: [batch, hidden]
        pooled = attended.mean(dim=1)

        # Predict for each target asset
        predictions = {}
        for asset in self.config.TARGET_ASSETS:
            pred = self.classifiers[asset](pooled)  # [batch, 1]
            predictions[asset] = pred

        return predictions, attention_weights

    def get_attention_weights_per_asset(self, attention_weights):
        """
        Reshape attention weights to show asset-level interactions.

        Args:
            attention_weights: [batch, seq_len*num_assets, seq_len*num_assets]

        Returns:
            asset_attention: [batch, num_assets, num_assets]
        """
        batch_size, total_len, _ = attention_weights.shape
        seq_len = total_len // len(self.config.ALL_ASSETS)
        num_assets = len(self.config.ALL_ASSETS)

        # Reshape to separate sequence and assets
        # [batch, num_assets, seq_len, num_assets, seq_len]
        reshaped = attention_weights.view(
            batch_size,
            num_assets, seq_len,
            num_assets, seq_len
        )

        # Average over time dimensions to get asset-to-asset attention
        asset_attention = reshaped.mean(dim=(2, 4))  # [batch, num_assets, num_assets]

        return asset_attention


if __name__ == "__main__":
    # Test model architecture
    config = Config()

    # Create dummy input
    batch_size = 8
    seq_len = config.SEQUENCE_LENGTH
    num_assets = len(config.ALL_ASSETS)
    features_per_asset = 14
    total_features = num_assets * features_per_asset

    x = torch.randn(batch_size, seq_len, total_features)

    # Initialize model
    model = MultiAssetDirectionalModel(config)

    print("="*60)
    print("MODEL ARCHITECTURE TEST")
    print("="*60)
    print(f"Input shape: {x.shape}")
    print(f"Sequence length: {seq_len}")
    print(f"Assets: {config.ALL_ASSETS}")
    print(f"Features per asset: {features_per_asset}")
    print(f"Total features: {total_features}")

    # Forward pass
    predictions, attention_weights = model(x)

    print("\nPredictions:")
    for asset, pred in predictions.items():
        print(f"  {asset:12s}: {pred.shape}")

    print(f"\nAttention weights shape: {attention_weights.shape}")

    # Test asset-level attention
    asset_attn = model.get_attention_weights_per_asset(attention_weights)
    print(f"Asset-to-asset attention: {asset_attn.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    print("\n✓ Model architecture test passed!")
