"""
TCN with Variable Selection Network (VSN)

Combines:
- VSN from TFT (automatic feature selection)
- TCN architecture (dilated convolutions for multi-scale patterns)
- Cross-asset gating (asset relationships)

Should achieve positive returns through VSN noise reduction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.variable_selection import SimpleVariableSelection


class TemporalBlock(nn.Module):
    """Temporal block with dilated causal convolution."""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.3):
        super(TemporalBlock, self).__init__()

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

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = out[:, :, :-self.padding] if self.padding != 0 else out
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = out[:, :, :-self.padding] if self.padding != 0 else out
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    """Temporal Convolutional Network with stacked blocks."""
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.3):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation = 2 ** i
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
        # Expect (batch, features, seq_len)
        return self.network(x)


class CrossAssetGating(nn.Module):
    """Gated mechanism for cross-asset information flow."""
    def __init__(self, hidden_size, num_assets):
        super(CrossAssetGating, self).__init__()
        self.num_assets = num_assets

        self.asset_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size * num_assets, hidden_size),
                nn.Sigmoid()
            )
            for _ in range(num_assets)
        ])

        self.combiners = nn.ModuleList([
            nn.Linear(hidden_size * num_assets, hidden_size)
            for _ in range(num_assets)
        ])

    def forward(self, asset_features):
        all_features = torch.cat(asset_features, dim=-1)

        gated = []
        for i in range(self.num_assets):
            gate = self.asset_gates[i](all_features)
            combined = self.combiners[i](all_features)
            gated_feature = gate * combined + (1 - gate) * asset_features[i]
            gated.append(gated_feature)

        return gated


class TCNWithVSN(nn.Module):
    """
    TCN with Variable Selection Network.

    Architecture:
    1. Per-asset VSN (learns important features)
    2. Per-asset TCN encoders (multi-scale temporal patterns)
    3. Cross-asset gating (asset relationships)
    4. Asset-specific prediction heads

    VSN eliminates noise, TCN captures patterns, gating models relationships.
    """
    def __init__(self, config):
        super(TCNWithVSN, self).__init__()
        self.config = config

        input_dim = config.INPUT_DIM  # 219
        hidden_size = config.LSTM_HIDDEN_SIZE  # 128 (reuse param name)
        dropout = config.DROPOUT  # 0.4

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

        # Per-asset TCN encoders
        # 6 levels: receptive field = 2^6 * 3 = 192 timesteps
        num_channels = [hidden_size] * 6

        self.asset_tcns = nn.ModuleList([
            TCN(
                num_inputs=hidden_size,  # VSN output
                num_channels=num_channels,
                kernel_size=3,
                dropout=dropout
            )
            for _ in config.TARGET_ASSETS
        ])

        # Cross-asset gating
        self.cross_asset_gating = CrossAssetGating(
            hidden_size=hidden_size,
            num_assets=len(config.TARGET_ASSETS)
        )

        # Prediction heads
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
        """
        batch_size, seq_len, input_dim = x.shape
        predictions = {}
        asset_features = []

        for i, asset in enumerate(self.config.TARGET_ASSETS):
            # 1. Variable Selection (KEY: noise reduction)
            selected, weights = self.vsn_networks[asset](x)  # (batch, seq_len, hidden_size)

            # 2. TCN encoding
            # TCN expects (batch, features, seq_len)
            tcn_input = selected.transpose(1, 2)
            tcn_out = self.asset_tcns[i](tcn_input)  # (batch, hidden_size, seq_len)

            # Back to (batch, seq_len, hidden_size)
            tcn_out = tcn_out.transpose(1, 2)

            # Global average pooling
            pooled = tcn_out.mean(dim=1)  # (batch, hidden_size)
            asset_features.append(pooled)

        # 3. Cross-asset gating
        gated_features = self.cross_asset_gating(asset_features)

        # 4. Predictions
        for i, asset in enumerate(self.config.TARGET_ASSETS):
            pred = self.classifiers[asset](gated_features[i])
            predictions[asset] = pred

        return predictions


if __name__ == "__main__":
    # Test the model
    from src.utils.config import Config

    config = Config()
    config.INPUT_DIM = 219
    config.LSTM_HIDDEN_SIZE = 128
    config.DROPOUT = 0.4

    model = TCNWithVSN(config)

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
    print("✓ TCN with VSN Test Passed!")
