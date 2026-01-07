# src/evaluation/metrics.py
"""
Evaluation metrics for multi-asset directional prediction
"""
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import Config

class Evaluator:
    """Evaluate model performance"""

    def __init__(self, model, test_loader, config=None):
        self.config = config if config else Config()
        self.model = model.to(self.config.DEVICE)
        self.test_loader = test_loader

    def evaluate(self):
        """Comprehensive evaluation"""
        self.model.eval()

        # Collect predictions and labels
        all_predictions = {asset: [] for asset in self.config.TARGET_ASSETS}
        all_labels = {asset: [] for asset in self.config.TARGET_ASSETS}
        all_probs = {asset: [] for asset in self.config.TARGET_ASSETS}

        with torch.no_grad():
            for features, labels in self.test_loader:
                features = features.to(self.config.DEVICE)

                # Forward pass
                predictions, _ = self.model(features)

                # Store predictions
                for asset in self.config.TARGET_ASSETS:
                    # Model outputs logits - apply sigmoid to get probabilities
                    logits = predictions[asset].cpu()
                    probs = torch.sigmoid(logits).numpy().flatten()
                    preds = (probs > 0.5).astype(int)
                    true_labels = labels[asset].numpy().flatten()

                    all_probs[asset].extend(probs)
                    all_predictions[asset].extend(preds)
                    all_labels[asset].extend(true_labels)

        # Convert to numpy arrays
        for asset in self.config.TARGET_ASSETS:
            all_predictions[asset] = np.array(all_predictions[asset])
            all_labels[asset] = np.array(all_labels[asset])
            all_probs[asset] = np.array(all_probs[asset])

        # Print prediction distribution (diagnostic)
        print("\nPrediction Distribution (diagnostic):")
        for asset in self.config.TARGET_ASSETS:
            preds = all_predictions[asset]
            probs = all_probs[asset]
            n_pos = (preds == 1).sum()
            n_neg = (preds == 0).sum()
            print(f"  {asset}: Pred 0={n_neg} ({n_neg/(n_pos+n_neg)*100:.1f}%), "
                  f"Pred 1={n_pos} ({n_pos/(n_pos+n_neg)*100:.1f}%), "
                  f"Prob range=[{probs.min():.3f}, {probs.max():.3f}]")

        # Compute metrics
        metrics = self.compute_metrics(all_labels, all_predictions, all_probs)

        return metrics, all_predictions, all_labels, all_probs

    def compute_metrics(self, labels, predictions, probabilities):
        """Compute all metrics"""
        metrics = {}

        for asset in self.config.TARGET_ASSETS:
            y_true = labels[asset]
            y_pred = predictions[asset]
            y_prob = probabilities[asset]

            metrics[asset] = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1': f1_score(y_true, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_true, y_prob),
                'confusion_matrix': confusion_matrix(y_true, y_pred)
            }

        # Overall metrics (average across assets)
        metrics['overall'] = {
            'accuracy': np.mean([metrics[asset]['accuracy'] for asset in self.config.TARGET_ASSETS]),
            'precision': np.mean([metrics[asset]['precision'] for asset in self.config.TARGET_ASSETS]),
            'recall': np.mean([metrics[asset]['recall'] for asset in self.config.TARGET_ASSETS]),
            'f1': np.mean([metrics[asset]['f1'] for asset in self.config.TARGET_ASSETS]),
            'roc_auc': np.mean([metrics[asset]['roc_auc'] for asset in self.config.TARGET_ASSETS])
        }

        return metrics

    def print_metrics(self, metrics):
        """Print formatted metrics"""
        print("\n" + "="*80)
        print("EVALUATION METRICS")
        print("="*80)

        # Per-asset metrics
        for asset in self.config.TARGET_ASSETS:
            m = metrics[asset]
            print(f"\n{asset}:")
            print(f"  Accuracy:  {m['accuracy']:.4f}")
            print(f"  Precision: {m['precision']:.4f}")
            print(f"  Recall:    {m['recall']:.4f}")
            print(f"  F1 Score:  {m['f1']:.4f}")
            print(f"  ROC-AUC:   {m['roc_auc']:.4f}")
            print(f"  Confusion Matrix:")
            print(f"    {m['confusion_matrix']}")

        # Overall metrics
        print(f"\nOVERALL (Average):")
        m = metrics['overall']
        print(f"  Accuracy:  {m['accuracy']:.4f}")
        print(f"  Precision: {m['precision']:.4f}")
        print(f"  Recall:    {m['recall']:.4f}")
        print(f"  F1 Score:  {m['f1']:.4f}")
        print(f"  ROC-AUC:   {m['roc_auc']:.4f}")

        print("="*80 + "\n")

    def save_metrics(self, metrics):
        """Save metrics to CSV"""
        self.config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        # Create DataFrame
        rows = []
        for asset in self.config.TARGET_ASSETS:
            rows.append({
                'Asset': asset,
                **metrics[asset]
            })

        # Remove confusion matrix (not CSV-friendly)
        for row in rows:
            row.pop('confusion_matrix', None)

        df = pd.DataFrame(rows)

        # Add overall row
        overall_row = {'Asset': 'OVERALL', **metrics['overall']}
        df = pd.concat([df, pd.DataFrame([overall_row])], ignore_index=True)

        # Save
        path = self.config.RESULTS_DIR / 'metrics.csv'
        df.to_csv(path, index=False)
        print(f"✓ Metrics saved to {path}")

        return df
