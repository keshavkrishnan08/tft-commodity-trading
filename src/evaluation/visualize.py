# src/evaluation/visualize.py
"""
Visualization for attention weights and model interpretability
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import Config

class AttentionVisualizer:
    """Visualize attention weights to understand asset relationships"""

    def __init__(self, model, test_loader, config=None):
        self.config = config if config else Config()
        self.model = model.to(self.config.DEVICE)
        self.test_loader = test_loader

    def extract_attention_weights(self, num_samples=100):
        """Extract attention weights from test set"""
        self.model.eval()

        all_attention_weights = []

        with torch.no_grad():
            for i, (features, labels) in enumerate(self.test_loader):
                if i >= num_samples:
                    break

                features = features.to(self.config.DEVICE)

                # Forward pass
                predictions, attention_weights = self.model(features)

                # Get asset-level attention
                asset_attention = self.model.get_attention_weights_per_asset(attention_weights)

                all_attention_weights.append(asset_attention.cpu().numpy())

        # Average across batches
        avg_attention = np.mean(np.concatenate(all_attention_weights, axis=0), axis=0)

        return avg_attention  # [num_assets, num_assets]

    def plot_asset_attention_heatmap(self, save_path=None):
        """Plot heatmap showing which assets attend to which"""
        print("\nExtracting attention weights...")
        attention = self.extract_attention_weights()

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot heatmap
        sns.heatmap(
            attention,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            xticklabels=self.config.ALL_ASSETS,
            yticklabels=self.config.ALL_ASSETS,
            cbar_kws={'label': 'Attention Weight'},
            ax=ax,
            vmin=0,
            vmax=1
        )

        ax.set_title('Multi-Asset Attention Matrix\n(Row → Column: Which assets influence which)',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Target Asset (Attended To)', fontsize=12)
        ax.set_ylabel('Source Asset (Attending From)', fontsize=12)

        plt.tight_layout()

        if save_path is None:
            save_path = self.config.PLOT_DIR / 'attention_heatmap.png'
            self.config.PLOT_DIR.mkdir(parents=True, exist_ok=True)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Attention heatmap saved to {save_path}")
        plt.close()

        return attention

    def analyze_leading_indicators(self, attention):
        """Identify which assets are leading indicators"""
        print("\n" + "="*80)
        print("ATTENTION ANALYSIS: Leading-Lag Relationships")
        print("="*80)

        # Sum attention given BY each asset (rows)
        attention_given = attention.sum(axis=1)

        # Sum attention received BY each asset (columns)
        attention_received = attention.sum(axis=0)

        # Create DataFrame for analysis
        import pandas as pd
        df = pd.DataFrame({
            'Asset': self.config.ALL_ASSETS,
            'Attention Given': attention_given,
            'Attention Received': attention_received,
            'Net Influence': attention_given - attention_received
        })

        df = df.sort_values('Net Influence', ascending=False)

        print("\nAsset Influence Ranking:")
        print(df.to_string(index=False))

        print("\nInterpretation:")
        print("  - Attention Given: How much this asset attends to others (outgoing influence)")
        print("  - Attention Received: How much others attend to this asset (incoming influence)")
        print("  - Net Influence: Positive = Leading indicator, Negative = Lagging indicator")

        # Identify strongest pairwise relationships
        print("\n" + "-"*80)
        print("Strongest Pairwise Relationships:")
        print("-"*80)

        # Get top 10 pairs (excluding self-attention)
        pairs = []
        for i, asset1 in enumerate(self.config.ALL_ASSETS):
            for j, asset2 in enumerate(self.config.ALL_ASSETS):
                if i != j:
                    pairs.append({
                        'From': asset1,
                        'To': asset2,
                        'Weight': attention[i, j]
                    })

        pairs_df = pd.DataFrame(pairs).sort_values('Weight', ascending=False)
        print(pairs_df.head(10).to_string(index=False))

        print("="*80 + "\n")

        return df

    def plot_training_history(self, history, save_path=None):
        """Plot training and validation metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Loss
        axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
        axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
        axes[0, 0].set_title('Loss Over Time', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('BCE Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Accuracy
        axes[0, 1].plot(history['train_acc'], label='Train Accuracy', linewidth=2)
        axes[0, 1].plot(history['val_acc'], label='Val Accuracy', linewidth=2)
        axes[0, 1].set_title('Accuracy Over Time', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Learning Rate
        axes[1, 0].plot(history['lr'], linewidth=2, color='red')
        axes[1, 0].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)

        # Overfitting gap
        gap = np.array(history['train_loss']) - np.array(history['val_loss'])
        axes[1, 1].plot(gap, linewidth=2, color='purple')
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].set_title('Overfitting Gap (Train - Val Loss)', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss Difference')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path is None:
            save_path = self.config.PLOT_DIR / 'training_history.png'
            self.config.PLOT_DIR.mkdir(parents=True, exist_ok=True)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training history saved to {save_path}")
        plt.close()

    def plot_confusion_matrices(self, metrics, save_path=None):
        """Plot confusion matrices for all assets"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()

        for i, asset in enumerate(self.config.TARGET_ASSETS):
            cm = metrics[asset]['confusion_matrix']

            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=['Down', 'Up'],
                yticklabels=['Down', 'Up'],
                ax=axes[i],
                cbar_kws={'label': 'Count'}
            )

            axes[i].set_title(f'{asset} Confusion Matrix', fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')

            # Add accuracy in title
            acc = metrics[asset]['accuracy']
            axes[i].text(0.5, -0.15, f'Accuracy: {acc:.3f}',
                        transform=axes[i].transAxes,
                        ha='center', fontsize=10)

        plt.tight_layout()

        if save_path is None:
            save_path = self.config.PLOT_DIR / 'confusion_matrices.png'
            self.config.PLOT_DIR.mkdir(parents=True, exist_ok=True)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrices saved to {save_path}")
        plt.close()
