# src/meta/manifold.py
"""
Latent Manifold Autoencoder for Market Stress Representation.
Compresses cross-asset features into a low-dimensional stress manifold.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config import Config

class DivergenceAE(nn.Module):
    """
    Autoencoder to learn the latent 'Market Stress Manifold'.
    """
    def __init__(self, input_dim, latent_dim=8, hidden_dims=[64, 32]):
        super(DivergenceAE, self).__init__()
        
        # Encoder
        encoder_layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(curr_dim, h_dim))
            encoder_layers.append(nn.BatchNorm1d(h_dim))
            encoder_layers.append(nn.LeakyReLU(0.2))
            encoder_layers.append(nn.Dropout(0.1))
            curr_dim = h_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        self.latent_proj = nn.Linear(curr_dim, latent_dim)
        
        # Decoder
        decoder_layers = []
        curr_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(curr_dim, h_dim))
            decoder_layers.append(nn.BatchNorm1d(h_dim))
            decoder_layers.append(nn.LeakyReLU(0.2))
            decoder_layers.append(nn.Dropout(0.1))
            curr_dim = h_dim
        
        decoder_layers.append(nn.Linear(curr_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        h = self.encoder(x)
        return self.latent_proj(h)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z

class ManifoldTrainer:
    """Helper to train the manifold autoencoder"""
    def __init__(self, input_dim, latent_dim=8, device='cpu'):
        self.device = device
        self.model = DivergenceAE(input_dim, latent_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()
        
    def train_step(self, x):
        self.model.train()
        self.optimizer.zero_grad()
        
        x = x.to(self.device)
        recon, z = self.model(x)
        loss = self.criterion(recon, x)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def get_latent(self, x):
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            return self.model.encode(x)

if __name__ == "__main__":
    # Test AE
    input_dim = 130  # Actual feature count from pipeline
    ae = DivergenceAE(input_dim=input_dim, latent_dim=8)
    
    test_input = torch.randn(16, input_dim)
    recon, z = ae(test_input)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Latent shape: {z.shape}")
    print(f"Reconstruction shape: {recon.shape}")
    print("✓ Divergence Manifold Autoencoder test passed!")
