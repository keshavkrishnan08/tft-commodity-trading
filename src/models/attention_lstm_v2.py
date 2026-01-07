# src/models/attention_lstm_v2.py
"""
Improved Multi-Asset Directional Prediction Model with:
1. Separate asset-level cross-attention
2. Attention entropy regularization
3. Higher dropout (0.5)
4. Better gradient flow
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import Config


class CrossAssetAttention(nn.Module):
    """
    Cross-asset attention mechanism.
    Each asset attends to all other assets to learn relationships.
    """
    def __init__(self, hidden_size, num_heads=8, dropout=0.5):
        super(CrossAssetAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        assert hidden_size % num_heads == 0

        self.head_dim = hidden_size // num_heads

        # Query, Key, Value projections
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        """
        Args:
            x: [batch, num_assets, hidden_size]

        Returns:
            output: [batch, num_assets, hidden_size]
            attention_weights: [batch, num_heads, num_assets, num_assets]
        """
        batch_size, num_assets, hidden_size = x.shape

        # Project to Q, K, V
        Q = self.q_proj(x)  # [batch, num_assets, hidden]
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape for multi-head: [batch, num_assets, num_heads, head_dim]
        Q = Q.view(batch_size, num_assets, self.num_heads, self.head_dim)
        K = K.view(batch_size, num_assets, self.num_heads, self.head_dim)
        V = V.view(batch_size, num_assets, self.num_heads, self.head_dim)

        # Transpose to [batch, num_heads, num_assets, head_dim]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # scores: [batch, num_heads, num_assets, num_assets]

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        # attended: [batch, num_heads, num_assets, head_dim]

        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous()
        # attended: [batch, num_assets, num_heads, head_dim]

        attended = attended.view(batch_size, num_assets, hidden_size)

        # Output projection
        output = self.out_proj(attended)

        # Residual + LayerNorm
        output = self.layer_norm(x + self.dropout(output))

        # Average attention weights across heads for interpretability
        attention_weights_avg = attention_weights.mean(dim=1)
        # [batch, num_assets, num_assets]

        return output, attention_weights_avg

    def compute_attention_entropy(self, attention_weights):
        """
        Compute entropy of attention distribution.
        High entropy = uniform attention (bad)
        Low entropy = focused attention (good)

        Args:
            attention_weights: [batch, num_assets, num_assets]

        Returns:
            entropy: scalar
        """
        # Add small epsilon to avoid log(0)
        eps = 1e-8
        attention_weights = attention_weights + eps

        # Compute entropy: -sum(p * log(p))
        entropy = -(attention_weights * torch.log(attention_weights)).sum(dim=-1).mean()

        return entropy


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


class ImprovedMultiAssetModel(nn.Module):
    """
    Improved Multi-Asset Directional Prediction Model.

    Key improvements:
        1. Cross-asset attention (not time-mixed)
        2. Attention entropy regularization
        3. Higher dropout (0.5)
        4. Better architecture for learning asset relationships
    """
    def __init__(self, config=None):
        super(ImprovedMultiAssetModel, self).__init__()

        self.config = config if config else Config()

        # Override dropout to 0.5
        dropout = 0.5

        # Number of features per asset
        self.features_per_asset = 14

        # LSTM encoders for each asset
        self.asset_lstms = nn.ModuleList([
            AssetLSTM(
                input_size=self.features_per_asset,
                hidden_size=self.config.LSTM_HIDDEN_SIZE,
                num_layers=self.config.LSTM_LAYERS,
                dropout=dropout
            )
            for _ in self.config.ALL_ASSETS
        ])

        # Cross-asset attention
        self.cross_asset_attention = CrossAssetAttention(
            hidden_size=self.config.LSTM_HIDDEN_SIZE,
            num_heads=self.config.ATTENTION_HEADS,
            dropout=dropout
        )

        # Classification heads (one per target asset)
        self.classifiers = nn.ModuleDict({
            asset: nn.Sequential(
                nn.Linear(self.config.LSTM_HIDDEN_SIZE, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
            for asset in self.config.TARGET_ASSETS
        })

    def forward(self, x, return_attention_entropy=False):
        """
        Args:
            x: [batch, seq_len, num_assets * features_per_asset]
            return_attention_entropy: If True, return entropy for regularization

        Returns:
            predictions: dict of {asset: probability [batch, 1]}
            attention_weights: [batch, num_assets, num_assets]
            attention_entropy: scalar (if return_attention_entropy=True)
        """
        batch_size, seq_len, total_features = x.shape

        # Process each asset through its own LSTM
        asset_outputs = []
        for i, asset in enumerate(self.config.ALL_ASSETS):
            # Extract features for this asset
            start_idx = i * self.features_per_asset
            end_idx = start_idx + self.features_per_asset
            asset_features = x[:, :, start_idx:end_idx]

            # LSTM encoding
            lstm_out = self.asset_lstms[i](asset_features)
            # [batch, seq_len, hidden]

            # Take last timestep as asset representation
            asset_repr = lstm_out[:, -1, :]  # [batch, hidden]
            asset_outputs.append(asset_repr)

        # Stack asset representations: [batch, num_assets, hidden]
        stacked_assets = torch.stack(asset_outputs, dim=1)

        # Cross-asset attention
        attended, attention_weights = self.cross_asset_attention(stacked_assets)
        # attended: [batch, num_assets, hidden]
        # attention_weights: [batch, num_assets, num_assets]

        # Predict for each target asset
        predictions = {}
        for i, asset in enumerate(self.config.TARGET_ASSETS):
            # Use attended representation for this asset
            asset_attended = attended[:, i, :]  # [batch, hidden]
            pred = self.classifiers[asset](asset_attended)
            predictions[asset] = pred

        if return_attention_entropy:
            attention_entropy = self.cross_asset_attention.compute_attention_entropy(attention_weights)
            return predictions, attention_weights, attention_entropy
        else:
            return predictions, attention_weights

    def get_attention_weights_per_asset(self, attention_weights):
        """
        Attention weights are already at asset level.
        Just return the average across batch.

        Args:
            attention_weights: [batch, num_assets, num_assets]

        Returns:
            asset_attention: [batch, num_assets, num_assets]
        """
        return attention_weights


if __name__ == "__main__":
    # Test model
    config = Config()
    model = ImprovedMultiAssetModel(config)

    # Create dummy input
    batch_size = 32
    seq_len = config.SEQUENCE_LENGTH
    num_assets = len(config.ALL_ASSETS)
    features_per_asset = 14
    total_features = num_assets * features_per_asset

    x = torch.randn(batch_size, seq_len, total_features)

    # Forward pass
    predictions, attention_weights, entropy = model(x, return_attention_entropy=True)

    print("="*80)
    print("IMPROVED MODEL ARCHITECTURE TEST")
    print("="*80)
    print(f"Input shape: {x.shape}")
    print(f"\nPredictions:")
    for asset, pred in predictions.items():
        print(f"  {asset}: {pred.shape}")
    print(f"\nAttention weights shape: {attention_weights.shape}")
    print(f"Attention entropy: {entropy.item():.4f}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print("="*80)
