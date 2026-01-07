# src/meta/policy.py
"""
MARS-Meta Policy Network.
Optimizes execution parameters using differentiable soft backtesting.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

class SoftBacktestLoss(nn.Module):
    """
    Differentiable P&L loss for direct policy optimization.
    """
    def __init__(self, risk_aversion=0.5):
        super(SoftBacktestLoss, self).__init__()
        self.risk_aversion = risk_aversion

    def forward(self, action_logits, directional_probs, future_returns):
        """
        Args:
            action_logits: [batch, 4] -> (Entry, SL, TP, Size)
            directional_probs: [batch, 1] -> Stage 1 output
            future_returns: [batch, 1] -> Ground truth future return
        
        Returns:
            neg_pnl: Negative Expected P&L (to be minimized)
        """
        # Map logits to actual parameters
        # Entry: offset from current price [0, 0.02]
        entry = torch.sigmoid(action_logits[:, 0:1]) * 0.02
        # SL: distance [0.01, 0.05]
        sl = 0.01 + torch.sigmoid(action_logits[:, 1:2]) * 0.04
        # TP: distance [0.02, 0.10]
        tp = 0.02 + torch.sigmoid(action_logits[:, 2:3]) * 0.08
        # Size: leverage [0, 1]
        size = torch.sigmoid(action_logits[:, 3:4])

        # Direction: 1 if prob > 0.5, else -1
        side = torch.sgn(directional_probs - 0.5)

        # Soft Expected P&L
        # P&L = Size * (Return * Side) if not stopped out
        # Here we use a 'soft' version of the logic for differentiability
        raw_pnl = size * (future_returns * side)
        
        # Penalize large SL and small TP ratio (Risk/Reward)
        rr_penalty = sl / (tp + 1e-6)
        
        # Risk adj return
        expected_pnl = torch.mean(raw_pnl - self.risk_aversion * rr_penalty)
        
        return -expected_pnl

class MetaPolicyNN(nn.Module):
    """
    Policy Network that maps latent market state + Stage 1 predictions 
    to execution parameters.
    """
    def __init__(self, latent_dim, num_assets=4):
        super(MetaPolicyNN, self).__init__()
        
        # Input: [Manifold Latent Vector (latent_dim)] + [Stage 1 Probs (num_assets)]
        input_dim = latent_dim + num_assets
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_assets * 4) # (Entry, SL, TP, Size) per asset
        )

    def forward(self, latent_z, stage1_probs):
        """
        Args:
            latent_z: [batch, latent_dim]
            stage1_probs: [batch, num_assets]
        """
        x = torch.cat([latent_z, stage1_probs], dim=1)
        params = self.network(x)
        
        # Reshape to [batch, num_assets, 4]
        return params.view(-1, 4, 4) if params.size(1) == 16 else params # Dynamic reshape hack

if __name__ == "__main__":
    # Test Policy Network
    model = MetaPolicyNN(latent_dim=8, num_assets=4)
    criterion = SoftBacktestLoss()
    
    latent_z = torch.randn(8, 8)
    probs = torch.rand(8, 4)
    future_rets = torch.randn(8, 1)
    
    actions = model(latent_z, probs)
    print(f"Action params shape: {actions.shape}")
    
    # Test loss for first asset
    loss = criterion(actions[:, 0, :], probs[:, 0:1], future_rets)
    print(f"Sample Loss: {loss.item():.4f}")
    print("✓ MARS-Meta Policy Network test passed!")
