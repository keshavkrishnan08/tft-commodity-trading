"""
Variable Selection Network (VSN)

Learns to select and weight important features from high-dimensional input.
Based on TFT architecture but standalone for use with any model.

Key components:
- Per-feature Gated Residual Networks (GRNs)
- Softmax-based feature selection
- Residual connections
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedLinearUnit(nn.Module):
    """GLU for controlling information flow."""
    def __init__(self, input_dim, output_dim=None, dropout=0.1):
        super(GatedLinearUnit, self).__init__()
        if output_dim is None:
            output_dim = input_dim
        self.output_dim = output_dim
        self.fc = nn.Linear(input_dim, output_dim * 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc(x)
        x = self.dropout(x)
        return F.glu(x, dim=-1)


class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN) for feature processing.

    Components:
    - Layer normalization
    - Dense layer
    - ELU activation
    - Gating mechanism
    - Residual connection
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super(GatedResidualNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.layer_norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.gate = GatedLinearUnit(output_dim, dropout=dropout)

        # Project input to output dimension if needed
        if input_dim != output_dim:
            self.skip_layer = nn.Linear(input_dim, output_dim)
        else:
            self.skip_layer = None

    def forward(self, x):
        # Layer norm
        residual = x
        x = self.layer_norm(x)

        # Dense layers with ELU
        x = self.fc1(x)
        x = self.elu(x)
        x = self.fc2(x)
        x = self.dropout(x)

        # Gating
        x = self.gate(x)

        # Residual connection
        if self.skip_layer is not None:
            residual = self.skip_layer(residual)
        return x + residual


class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network for learning feature importance.

    Architecture:
    1. Per-feature GRNs (process each feature independently)
    2. Flatten and aggregate
    3. Selection GRN (learn feature weights)
    4. Softmax to get importance weights
    5. Apply weights to features

    Benefits:
    - Automatic feature selection
    - Interpretable feature importance
    - Reduces overfitting by focusing on relevant features
    """
    def __init__(self, input_dim, num_features, hidden_dim, dropout=0.1):
        """
        Args:
            input_dim: Dimension of each input feature (usually 1 for raw features)
            num_features: Number of input features
            hidden_dim: Hidden dimension for GRNs
            dropout: Dropout rate
        """
        super(VariableSelectionNetwork, self).__init__()
        self.input_dim = input_dim
        self.num_features = num_features
        self.hidden_dim = hidden_dim

        # Per-feature GRNs (process each feature independently)
        self.feature_grns = nn.ModuleList([
            GatedResidualNetwork(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                dropout=dropout
            )
            for _ in range(num_features)
        ])

        # Variable selection GRN (learns which features are important)
        self.selection_grn = GatedResidualNetwork(
            input_dim=num_features * hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=num_features,
            dropout=dropout
        )

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, num_features) or (batch, num_features)

        Returns:
            selected_features: (batch, [seq_len,] num_features * hidden_dim)
            feature_weights: (batch, [seq_len,] num_features) - interpretable weights
        """
        # Handle 2D (batch, features) and 3D (batch, seq, features) inputs
        original_shape = x.shape
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # (batch, 1, features)

        batch_size, seq_len, num_features = x.shape

        # Process each feature independently through its GRN
        processed_features = []
        for i in range(self.num_features):
            # Extract feature i: (batch, seq_len, 1)
            feature = x[:, :, i:i+1]

            # Process through GRN: (batch, seq_len, hidden_dim)
            processed = self.feature_grns[i](feature)
            processed_features.append(processed)

        # Stack: (batch, seq_len, num_features, hidden_dim)
        stacked = torch.stack(processed_features, dim=2)

        # Flatten for selection: (batch, seq_len, num_features * hidden_dim)
        flattened = stacked.reshape(batch_size, seq_len, -1)

        # Learn feature importance weights: (batch, seq_len, num_features)
        weights_logits = self.selection_grn(flattened)

        # Softmax for normalized weights
        feature_weights = F.softmax(weights_logits, dim=-1)

        # Apply weights to processed features
        # feature_weights: (batch, seq_len, num_features, 1)
        weighted = stacked * feature_weights.unsqueeze(-1)

        # Sum over features: (batch, seq_len, hidden_dim)
        selected = weighted.sum(dim=2)

        # Restore original shape
        if len(original_shape) == 2:
            selected = selected.squeeze(1)  # (batch, hidden_dim)
            feature_weights = feature_weights.squeeze(1)  # (batch, num_features)

        return selected, feature_weights


class SimpleVariableSelection(nn.Module):
    """
    Simplified Variable Selection for faster training.
    Uses single shared network instead of per-feature GRNs.
    """
    def __init__(self, num_features, hidden_dim, output_dim, dropout=0.1):
        super(SimpleVariableSelection, self).__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim

        # Feature processing
        self.feature_transform = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout)
        )

        # Feature importance learning
        self.importance_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_features)
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Args:
            x: (batch, [seq_len,] num_features)

        Returns:
            selected: (batch, [seq_len,] output_dim)
            weights: (batch, [seq_len,] num_features)
        """
        # Transform features
        h = self.feature_transform(x)

        # Learn importance weights
        weights_logits = self.importance_net(h)
        weights = F.softmax(weights_logits, dim=-1)

        # Apply weights to input features
        weighted_input = x * weights

        # Transform weighted input
        h_weighted = self.feature_transform(weighted_input)

        # Project to output dimension
        selected = self.output_proj(h_weighted)

        return selected, weights


if __name__ == "__main__":
    # Test VSN
    batch_size = 32
    seq_len = 60
    num_features = 219
    hidden_dim = 128

    # Test full VSN (slower but more expressive)
    vsn = VariableSelectionNetwork(
        input_dim=1,
        num_features=num_features,
        hidden_dim=hidden_dim,
        dropout=0.1
    )

    # Test 3D input (with sequence)
    x_3d = torch.randn(batch_size, seq_len, num_features)
    selected_3d, weights_3d = vsn(x_3d)
    print(f"3D Input: {x_3d.shape}")
    print(f"3D Selected: {selected_3d.shape}")
    print(f"3D Weights: {weights_3d.shape}")

    # Test 2D input (without sequence)
    x_2d = torch.randn(batch_size, num_features)
    selected_2d, weights_2d = vsn(x_2d)
    print(f"\n2D Input: {x_2d.shape}")
    print(f"2D Selected: {selected_2d.shape}")
    print(f"2D Weights: {weights_2d.shape}")

    # Test SimpleVSN (faster)
    simple_vsn = SimpleVariableSelection(
        num_features=num_features,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        dropout=0.1
    )

    selected_simple, weights_simple = simple_vsn(x_2d)
    print(f"\nSimple VSN Selected: {selected_simple.shape}")
    print(f"Simple VSN Weights: {weights_simple.shape}")

    print("\n✓ Variable Selection Network Test Passed!")
