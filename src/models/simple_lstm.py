"""
Simple LSTM baseline for comparison with TFT.
Uses the same 199-feature input as TFT but with basic LSTM architecture.
"""
import torch
import torch.nn as nn


class SimpleLSTM(nn.Module):
    """
    Basic LSTM model for multi-asset directional prediction.

    Architecture:
        - Shared LSTM encoder for all features
        - Asset-specific prediction heads
        - No attention mechanism (pure LSTM baseline)
    """
    def __init__(self, config):
        super(SimpleLSTM, self).__init__()

        self.config = config

        # Shared LSTM encoder
        self.lstm = nn.LSTM(
            input_size=config.INPUT_DIM,
            hidden_size=config.LSTM_HIDDEN_SIZE,
            num_layers=config.LSTM_LAYERS,
            batch_first=True,
            dropout=config.DROPOUT if config.LSTM_LAYERS > 1 else 0.0
        )

        # Dropout after LSTM
        self.dropout = nn.Dropout(config.DROPOUT)

        # Asset-specific prediction heads
        self.heads = nn.ModuleDict({
            asset: nn.Sequential(
                nn.Linear(config.LSTM_HIDDEN_SIZE, config.LSTM_HIDDEN_SIZE // 2),
                nn.ReLU(),
                nn.Dropout(config.DROPOUT),
                nn.Linear(config.LSTM_HIDDEN_SIZE // 2, 1)
            )
            for asset in config.TARGET_ASSETS
        })

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim) tensor

        Returns:
            dict of {asset: (batch, 1) logits}
        """
        # LSTM encoding
        # x shape: (batch, seq_len, input_dim)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Take last timestep output
        # lstm_out shape: (batch, seq_len, hidden_size)
        last_output = lstm_out[:, -1, :]  # (batch, hidden_size)

        # Apply dropout
        encoded = self.dropout(last_output)

        # Asset-specific predictions
        outputs = {}
        for asset in self.config.TARGET_ASSETS:
            outputs[asset] = self.heads[asset](encoded)

        return outputs


if __name__ == "__main__":
    # Test the model
    from src.utils.config import Config

    config = Config()
    config.INPUT_DIM = 199
    config.LSTM_HIDDEN_SIZE = 128
    config.LSTM_LAYERS = 2
    config.DROPOUT = 0.5

    model = SimpleLSTM(config)

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
    print("✓ SimpleLSTM Test Passed!")
