# src/models/temporal_fusion_transformer.py
"""
Temporal Fusion Transformer (TFT) for Multi-Asset Energy Trading.
Implements:
- Gated Residual Networks (GRN)
- Variable Selection Networks (VSN)
- Interpretable Multi-Head Attention with CAUSAL MASK
- Multi-Asset Output Heads
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
    # Gated Residual Network - the core building block of TFT
    # Uses skip connections to keep gradients flowing and gating to control what information passes through
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1, context_dim=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Main transformation pathway
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.context_fc = nn.Linear(context_dim, hidden_dim, bias=False) if context_dim else None
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.glu = GatedLinearUnit(hidden_dim, output_dim)
        # Skip connection helps with gradient flow - if dimensions don't match, we project
        self.skip_layer = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, context=None):
        # Save input for skip connection
        residual = self.skip_layer(x) if self.skip_layer else x
        # Main transformation with ELU activation
        hidden = F.elu(self.fc1(x))
        # Add context information if available (like asset type)
        if self.context_fc is not None and context is not None:
            hidden = hidden + self.context_fc(context)
        hidden = F.elu(self.fc2(hidden))
        hidden = self.dropout(hidden)
        # Gating controls what information flows through
        gated_output = self.glu(hidden)
        # Add skip connection and normalize for stable training
        return self.layer_norm(gated_output + residual)


class VariableSelectionNetwork(nn.Module):
    # Variable Selection Network - figures out which of the 199 features actually matter
    # Not all features are useful all the time. VSN adaptively picks the most relevant ones
    # based on current market conditions, reducing from 199 to ~20 effective features
    # We use a shared embedding + single GRN for efficiency instead of 199 separate GRNs
    def __init__(self, input_dim, num_features, hidden_dim, dropout=0.1, context_dim=None):
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        
        # Optimized: Single shared linear layer for feature embedding
        # Projects each feature's scalar value to a hidden vector
        self.feature_embedding = nn.Linear(1, hidden_dim)
        
        # Flattened GRN for variable selection weights
        self.selection_grn = GatedResidualNetwork(
            input_dim, hidden_dim, num_features, dropout, context_dim
        )
        
        # Final projection
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, context=None):
        batch_size, seq_len, num_features = x.shape
        # Step 1: Compute selection weights - which features get attention?
        flat_x = x.view(batch_size * seq_len, -1)
        flat_context = context.view(batch_size * seq_len, -1) if context is not None else None
        weights = self.selection_grn(flat_x, flat_context)
        weights = F.softmax(weights, dim=-1)  # Normalize to probabilities
        weights = weights.view(batch_size, seq_len, num_features, 1)
        # Step 2: Embed each feature value into a vector
        x_expanded = x.unsqueeze(-1)
        processed_features = self.feature_embedding(x_expanded)
        # Step 3: Weighted combination - features with high weights contribute more
        selected = (processed_features * weights).sum(dim=2)
        return self.output_projection(selected), weights.squeeze(-1)


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
    # Multi-head attention with causal masking - can't cheat by looking at the future!
    # At time t, we only look at information from times <= t (no look-ahead bias)
    # The attention weights are interpretable - they show which past days matter most
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
        
        # Compute attention scores (how similar is query to each key?)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # Causal mask: block future positions (can't use tomorrow's data today!)
        if mask is None:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=query.device), diagonal=1).bool()
            mask = mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask, float('-inf'))
        # Convert scores to probabilities
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        # Weighted combination of values
        context = torch.matmul(attn_weights, V)
        # Reshape back and project
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_linear(context)
        return output, attn_weights


class TemporalFusionTransformer(nn.Module):
    """
    Full Temporal Fusion Transformer for Multi-Asset Energy Trading.
    
    Architecture:
    1. Variable Selection Network (learns feature importance)
    2. Positional Encoding (temporal awareness)
    3. Multi-Head Self-Attention with CAUSAL MASK
    4. Gated Residual Networks (information gating)
    5. Multi-Asset Output Heads
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
        
        # Variable Selection Network (for 199 features)
        self.vsn = VariableSelectionNetwork(
            input_dim=self.input_dim,
            num_features=self.input_dim,  # Treating each input scalar as a feature
            hidden_dim=self.hidden_dim,
            dropout=self.dropout
        )
        
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
            x: (batch, seq_len, input_dim) - Input features
            time_features: (batch, seq_len, 3) - [day, month, week]
            
        Returns:
            outputs: dict of {asset: (batch, 1)} logits
            attention_weights: (batch, num_heads, seq, seq) from last layer
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. Variable Selection (Prioritization)
        # We need to reshape x to (batch, seq, num_features, 1) if VSN expects it
        # But our VSN is currently built for flattened inputs. 
        # Let's use the VSN to weight the inputs.
        selected_features, vsn_weights = self.vsn(x)  # (batch, seq, hidden)
        
        # 2. Time Embedding
        time_emb = self.time_projection(time_features)  # (batch, seq, hidden)
        
        # 3. Combine and Project
        hidden = selected_features + time_emb
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

    def get_vsn_weights(self, x):
        """
        Return Variable Selection Network weights for analysis.
        Args:
            x: (batch, seq, num_features)
        Returns:
            vsn_weights: (batch, seq, num_features)
        """
        _, vsn_weights = self.vsn(x)
        return vsn_weights


if __name__ == "__main__":
    # Test the TFT model
    from src.utils.config import Config
    config = Config()
    config.INPUT_DIM = 199
    config.TFT_HIDDEN_DIM = 128
    config.TFT_NUM_HEADS = 4
    config.TFT_NUM_LAYERS = 2
    config.SEQUENCE_LENGTH = 60
    
    model = TemporalFusionTransformer(config)
    print(f"TFT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    x = torch.randn(8, 60, 199)  # (batch, seq, features)
    t = torch.randn(8, 60, 3)    # (batch, seq, time_features)
    outputs, attn = model(x, t)
    
    for asset, out in outputs.items():
        print(f"{asset}: {out.shape}")
    print(f"Attention shape: {attn.shape}")
    print("✓ TFT Model Test Passed!")
