"""
Improved TCN with Variable Selection Network (V2)

Key improvements over V1:
1. Simpler architecture (removed complex cross-asset gating)
2. Better receptive field alignment (4 levels for 60-step sequences)
3. Stronger regularization
4. Residual connections for gradient flow
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.variable_selection import SimpleVariableSelection


class TemporalBlock(nn.Module):
    """Temporal block with dilated causal convolution."""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.4):
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

        # Ensure same length
        if res.size(2) != out.size(2):
            res = res[:, :, :out.size(2)]

        return self.relu(out + res)


class TCN(nn.Module):
    """Temporal Convolutional Network with stacked blocks."""
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.4):
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


class ImprovedTCNWithVSN(nn.Module):
    """
    Simplified TCN with Variable Selection Network (V2).

    Improvements:
    1. Removed cross-asset gating (was causing instability)
    2. Reduced TCN depth to 4 levels (better for 60-step sequences)
    3. Stronger dropout and weight decay
    4. Simpler prediction heads
    5. Better receptive field: 2^4 * 3 = 48 timesteps (fits 60-step input)
    """
    def __init__(self, config):
        super(ImprovedTCNWithVSN, self).__init__()
        self.config = config

        input_dim = config.INPUT_DIM  # 219
        hidden_size = config.LSTM_HIDDEN_SIZE  # 96 (reduced capacity)
        dropout = config.DROPOUT  # 0.4 (increased)

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
        # 4 levels: receptive field = 2^4 * 3 = 48 timesteps (good for 60-step input)
        num_channels = [hidden_size] * 4  # Reduced from 6 levels

        self.asset_tcns = nn.ModuleList([
            TCN(
                num_inputs=hidden_size,  # VSN output
                num_channels=num_channels,
                kernel_size=3,
                dropout=dropout
            )
            for _ in config.TARGET_ASSETS
        ])

        # Simplified prediction heads (no cross-asset interaction)
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

        for i, asset in enumerate(self.config.TARGET_ASSETS):
            # 1. Variable Selection (KEY: noise reduction)
            selected, weights = self.vsn_networks[asset](x)  # (batch, seq_len, hidden_size)

            # 2. TCN encoding
            # TCN expects (batch, features, seq_len)
            tcn_input = selected.transpose(1, 2)
            tcn_out = self.asset_tcns[i](tcn_input)  # (batch, hidden_size, seq_len)

            # Back to (batch, seq_len, hidden_size)
            tcn_out = tcn_out.transpose(1, 2)

            # Global max pooling (more robust than mean for TCN)
            pooled = tcn_out.max(dim=1)[0]  # (batch, hidden_size)

            # 3. Prediction
            pred = self.classifiers[asset](pooled)
            predictions[asset] = pred

        return predictions


if __name__ == "__main__":
    # Test the improved model
    from src.utils.config import Config

    config = Config()
    config.INPUT_DIM = 219
    config.LSTM_HIDDEN_SIZE = 96
    config.DROPOUT = 0.4

    model = ImprovedTCNWithVSN(config)

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
    print("✓ Improved TCN with VSN Test Passed!")
