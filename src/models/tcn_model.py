"""
Temporal Convolutional Network (TCN) for Multi-Asset Trading

Architecturally different from LSTM and TFT:
- Uses dilated causal convolutions instead of recurrent layers
- Residual connections for deep networks
- Handles long sequences efficiently with exponential receptive field growth
- Better at capturing multi-scale temporal patterns
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalBlock(nn.Module):
    """
    Single temporal block with dilated causal convolution.

    Key features:
    - Causal padding (no future information leakage)
    - Dilated convolutions (exponentially growing receptive field)
    - Residual connections (gradient flow)
    - Weight normalization (training stability)
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.3):
        super(TemporalBlock, self).__init__()

        # Causal padding: only look at past
        self.padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=self.padding, dilation=dilation
        )
        self.conv1 = nn.utils.weight_norm(self.conv1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=self.padding, dilation=dilation
        )
        self.conv2 = nn.utils.weight_norm(self.conv2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Residual connection
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x: (batch, channels, seq_len)
        Returns:
            out: (batch, channels, seq_len)
        """
        # First conv block
        out = self.conv1(x)
        # Remove future padding
        out = out[:, :, :-self.padding] if self.padding != 0 else out
        out = self.relu1(out)
        out = self.dropout1(out)

        # Second conv block
        out = self.conv2(out)
        out = out[:, :, :-self.padding] if self.padding != 0 else out
        out = self.relu2(out)
        out = self.dropout2(out)

        # Residual connection
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    """
    Temporal Convolutional Network with stacked temporal blocks.

    Receptive field grows exponentially: 2^num_levels * kernel_size
    Example: 8 levels with kernel_size=3 gives receptive field of 768 timesteps
    """
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.3):
        """
        Args:
            num_inputs: Number of input features
            num_channels: List of channel sizes for each level
            kernel_size: Size of convolutional kernel
            dropout: Dropout rate
        """
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation = 2 ** i  # Exponentially increasing dilation
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]

            layers.append(
                TemporalBlock(
                    in_channels, out_channels, kernel_size,
                    stride=1, dilation=dilation, dropout=dropout
                )
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, features) - standard format
        Returns:
            out: (batch, seq_len, features) - standard format
        """
        # TCN expects (batch, features, seq_len)
        x = x.transpose(1, 2)
        out = self.network(x)
        # Return to (batch, seq_len, features)
        out = out.transpose(1, 2)
        return out


class CrossAssetGating(nn.Module):
    """
    Gated mechanism for cross-asset information flow.
    Each asset can selectively incorporate information from other assets.
    """
    def __init__(self, hidden_size, num_assets):
        super(CrossAssetGating, self).__init__()
        self.num_assets = num_assets

        # Learn which assets are relevant for each target asset
        self.asset_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size * num_assets, hidden_size),
                nn.Sigmoid()
            )
            for _ in range(num_assets)
        ])

        # Combine gated information
        self.combiners = nn.ModuleList([
            nn.Linear(hidden_size * num_assets, hidden_size)
            for _ in range(num_assets)
        ])

    def forward(self, asset_features):
        """
        Args:
            asset_features: List of (batch, hidden_size) for each asset
        Returns:
            gated_features: List of (batch, hidden_size) for each asset
        """
        # Concatenate all asset features
        all_features = torch.cat(asset_features, dim=-1)  # (batch, hidden_size * num_assets)

        gated = []
        for i in range(self.num_assets):
            # Compute gates
            gate = self.asset_gates[i](all_features)  # (batch, hidden_size)

            # Combine with gating
            combined = self.combiners[i](all_features)  # (batch, hidden_size)
            gated_feature = gate * combined + (1 - gate) * asset_features[i]
            gated.append(gated_feature)

        return gated


class MultiAssetTCN(nn.Module):
    """
    Multi-Asset TCN for directional prediction.

    Architecture:
    1. Per-asset TCN encoders (capture temporal patterns independently)
    2. Cross-asset gating (model asset relationships)
    3. Per-asset prediction heads

    Key differences from LSTM/TFT:
    - No recurrence (parallelizable)
    - Dilated convolutions (multi-scale patterns)
    - Gating instead of attention (more efficient)
    """
    def __init__(self, config):
        super(MultiAssetTCN, self).__init__()
        self.config = config

        # TCN parameters
        self.features_per_asset = 26  # Same as WeeklyPredictionModel
        hidden_size = config.LSTM_HIDDEN_SIZE  # Reuse config parameter

        # Per-asset TCN encoders
        # 8 levels: receptive field = 2^8 * 3 = 768 timesteps (more than 60 seq_len)
        num_channels = [hidden_size] * 8

        self.asset_tcns = nn.ModuleList([
            TCN(
                num_inputs=self.features_per_asset,
                num_channels=num_channels,
                kernel_size=3,
                dropout=config.DROPOUT
            )
            for _ in config.ALL_ASSETS
        ])

        # Cross-asset gating
        self.cross_asset_gating = CrossAssetGating(
            hidden_size=hidden_size,
            num_assets=len(config.ALL_ASSETS)
        )

        # Prediction heads
        self.classifiers = nn.ModuleDict({
            asset: nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(config.DROPOUT),
                nn.Linear(hidden_size // 2, hidden_size // 4),
                nn.ReLU(),
                nn.Dropout(config.DROPOUT),
                nn.Linear(hidden_size // 4, 1)
            )
            for asset in config.TARGET_ASSETS
        })

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, num_assets * features_per_asset)

        Returns:
            predictions: dict of {asset: logits (batch, 1)}
        """
        batch_size, seq_len, total_features = x.shape

        # Process each asset through its TCN
        asset_features = []
        for i, asset in enumerate(self.config.ALL_ASSETS):
            start_idx = i * self.features_per_asset
            end_idx = start_idx + self.features_per_asset
            asset_input = x[:, :, start_idx:end_idx]  # (batch, seq_len, features_per_asset)

            # TCN encoding
            tcn_out = self.asset_tcns[i](asset_input)  # (batch, seq_len, hidden_size)

            # Global average pooling over time
            pooled = tcn_out.mean(dim=1)  # (batch, hidden_size)
            asset_features.append(pooled)

        # Cross-asset gating
        gated_features = self.cross_asset_gating(asset_features)

        # Predictions
        predictions = {}
        for i, asset in enumerate(self.config.TARGET_ASSETS):
            pred = self.classifiers[asset](gated_features[i])
            predictions[asset] = pred

        return predictions


if __name__ == "__main__":
    # Test the TCN model
    from src.utils.config import Config

    config = Config()
    config.LSTM_HIDDEN_SIZE = 128
    config.DROPOUT = 0.3

    model = MultiAssetTCN(config)

    # Test forward pass
    batch_size = 32
    seq_len = 60
    num_features = 26 * len(config.ALL_ASSETS)

    x = torch.randn(batch_size, seq_len, num_features)
    outputs = model(x)

    print(f"Input shape: {x.shape}")
    for asset, out in outputs.items():
        print(f"{asset} output shape: {out.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print("✓ TCN Model Test Passed!")
