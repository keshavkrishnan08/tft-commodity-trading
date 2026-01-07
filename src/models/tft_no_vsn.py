# src/models/tft_no_vsn.py
"""
ABLATION TEST: Temporal Fusion Transformer WITHOUT Variable Selection Network
- Removes VSN to test its contribution
- Directly projects all 199 features to hidden dimension
- Keeps all other components: causal attention, GRN, position encoding
- Uses same training/execution as V8 sliding
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GatedLinearUnit(nn.Module):
    """Gated Linear Unit for controlling information flow."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.gate_fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x) * torch.sigmoid(self.gate_fc(x))


class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN) - Core TFT building block.
    Includes skip connections and gating to control information flow.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1, context_dim=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Main pathway
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.context_fc = nn.Linear(context_dim, hidden_dim, bias=False) if context_dim else None
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Gated Linear Unit
        self.glu = GatedLinearUnit(hidden_dim, output_dim)

        # Skip connection (project if dimensions differ)
        self.skip_layer = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None

        # Layer Norm
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None):
        # Skip connection
        residual = self.skip_layer(x) if self.skip_layer else x

        # Main pathway
        hidden = F.elu(self.fc1(x))

        # Add context if provided
        if self.context_fc is not None and context is not None:
            hidden = hidden + self.context_fc(context)

        hidden = F.elu(self.fc2(hidden))
        hidden = self.dropout(hidden)

        # Gated output
        gated_output = self.glu(hidden)

        # Add residual and normalize
        return self.layer_norm(gated_output + residual)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal awareness."""
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: (batch, seq, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class InterpretableMultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with CAUSAL MASK.
    Prevents future data leakage by masking future positions.
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, _ = query.shape

        # Linear projections
        Q = self.q_linear(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # CAUSAL MASK: Prevent attending to future positions
        if mask is None:
            # Create lower-triangular causal mask
            mask = torch.triu(torch.ones(seq_len, seq_len, device=query.device), diagonal=1).bool()
            mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, seq)

        scores = scores.masked_fill(mask, float('-inf'))

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        context = torch.matmul(attn_weights, V)

        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_linear(context)

        return output, attn_weights


class TFT_NoVSN(nn.Module):
    """
    ABLATION: Temporal Fusion Transformer WITHOUT Variable Selection Network

    Architecture:
    1. ❌ REMOVED: Variable Selection Network
    2. ✅ KEPT: Direct feature projection (all 199 features)
    3. ✅ KEPT: Positional Encoding (temporal awareness)
    4. ✅ KEPT: Multi-Head Self-Attention with CAUSAL MASK
    5. ✅ KEPT: Gated Residual Networks (information gating)
    6. ✅ KEPT: Multi-Asset Output Heads

    This tests whether feature selection contributes to performance.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Dimensions
        self.input_dim = config.INPUT_DIM  # 199 features
        self.hidden_dim = getattr(config, 'TFT_HIDDEN_DIM', 128)
        self.num_heads = getattr(config, 'TFT_NUM_HEADS', 4)
        self.num_layers = getattr(config, 'TFT_NUM_LAYERS', 2)
        self.dropout = getattr(config, 'TFT_DROPOUT', 0.1)
        self.target_assets = config.TARGET_ASSETS

        # ❌ REMOVED: Variable Selection Network
        # ✅ REPLACED WITH: Direct feature projection
        self.feature_projection = nn.Linear(self.input_dim, self.hidden_dim)

        # Time feature projection (3 features from dataset: day, month, week)
        self.time_projection = nn.Linear(3, self.hidden_dim)

        # Input projection
        self.input_projection = nn.Linear(self.hidden_dim, self.hidden_dim)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            self.hidden_dim,
            max_len=config.SEQUENCE_LENGTH + 10,
            dropout=self.dropout
        )

        # Transformer encoder layers with causal attention
        self.attention_layers = nn.ModuleList([
            InterpretableMultiHeadAttention(self.hidden_dim, self.num_heads, self.dropout)
            for _ in range(self.num_layers)
        ])

        # GRN after each attention layer
        self.grn_layers = nn.ModuleList([
            GatedResidualNetwork(self.hidden_dim, self.hidden_dim, self.hidden_dim, self.dropout)
            for _ in range(self.num_layers)
        ])

        # Layer norms
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_dim) for _ in range(self.num_layers)
        ])

        # Final GRN before output
        self.final_grn = GatedResidualNetwork(
            self.hidden_dim, self.hidden_dim, self.hidden_dim, self.dropout
        )

        # Multi-asset output heads
        self.output_heads = nn.ModuleDict({
            asset: nn.Linear(self.hidden_dim, 1)
            for asset in self.target_assets
        })

        # Store attention weights for interpretability
        self.attention_weights = None

    def forward(self, x, time_features):
        """
        Forward pass.

        Args:
            x: (batch, seq_len, input_dim) - Input features (199 features)
            time_features: (batch, seq_len, 3) - [day, month, week]

        Returns:
            outputs: dict of {asset: (batch, 1)} logits
            attention_weights: (batch, num_heads, seq, seq) from last layer
        """
        batch_size, seq_len, _ = x.shape

        # 1. ❌ NO Variable Selection - directly project all features
        feature_emb = self.feature_projection(x)  # (batch, seq, hidden)

        # 2. Time Embedding
        time_emb = self.time_projection(time_features)  # (batch, seq, hidden)

        # 3. Combine and Project
        hidden = feature_emb + time_emb
        hidden = self.input_projection(hidden)

        # 4. Add positional encoding
        hidden = self.positional_encoding(hidden)

        # Create causal mask once
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device), diagonal=1
        ).bool().unsqueeze(0).unsqueeze(0)

        # Apply transformer layers
        for i in range(self.num_layers):
            # Multi-head attention with causal mask
            attn_out, attn_weights = self.attention_layers[i](
                hidden, hidden, hidden, mask=causal_mask
            )

            # Residual connection + layer norm
            hidden = self.layer_norms[i](hidden + attn_out)

            # GRN with gating
            hidden = self.grn_layers[i](hidden)

        # Store attention weights from last layer for interpretability
        self.attention_weights = attn_weights

        # Final GRN
        hidden = self.final_grn(hidden)

        # Take the LAST timestep (prediction for T+1)
        final_hidden = hidden[:, -1, :]  # (batch, hidden)

        # Multi-asset outputs
        outputs = {}
        for asset in self.target_assets:
            outputs[asset] = self.output_heads[asset](final_hidden).squeeze(-1)

        return outputs, attn_weights

    def get_attention_weights(self):
        """Return stored attention weights for interpretability."""
        return self.attention_weights


if __name__ == "__main__":
    # Test the ablation model
    from src.utils.config import Config
    config = Config()
    config.INPUT_DIM = 199
    config.TFT_HIDDEN_DIM = 128
    config.TFT_NUM_HEADS = 4
    config.TFT_NUM_LAYERS = 2
    config.SEQUENCE_LENGTH = 60

    model = TFT_NoVSN(config)

    # Count parameters
    vsn_params = sum(p.numel() for p in model.parameters())
    print(f"TFT (No VSN) Parameters: {vsn_params:,}")

    # Test forward pass
    x = torch.randn(8, 60, 199)  # (batch, seq, features)
    t = torch.randn(8, 60, 3)    # (batch, seq, time_features)
    outputs, attn = model(x, t)

    for asset, out in outputs.items():
        print(f"{asset}: {out.shape}")
    print(f"Attention shape: {attn.shape}")
    print("✓ TFT (No VSN) Ablation Model Test Passed!")
