# src/training/ensemble_trainer.py
"""
Ensemble training: Train multiple models with different seeds
Combines predictions by averaging for improved accuracy
"""
import torch
import numpy as np
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.training.trainer import Trainer
from src.models.weekly_model import WeeklyPredictionModel
from src.utils.config import Config


class EnsembleTrainer:
    """Train and manage ensemble of models"""

    def __init__(self, config=None):
        self.config = config if config else Config()
        self.models = []
        self.trainers = []

    def train_ensemble(self, train_loader, val_loader):
        """
        Train ensemble of models with different random seeds.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader

        Returns:
            histories: List of training histories for each model
        """
        print("\n" + "="*80)
        print(f"TRAINING ENSEMBLE OF {self.config.ENSEMBLE_SIZE} MODELS")
        print("="*80)

        histories = []

        for i, seed in enumerate(self.config.ENSEMBLE_SEEDS[:self.config.ENSEMBLE_SIZE]):
            print(f"\n{'='*80}")
            print(f"ENSEMBLE MODEL {i+1}/{self.config.ENSEMBLE_SIZE} (seed={seed})")
            print("="*80)

            # Set random seed
            self._set_seed(seed)

            # Create new model
            model = WeeklyPredictionModel(self.config)

            # Create trainer
            trainer = Trainer(model, train_loader, val_loader, self.config)

            # Train
            history = trainer.fit()

            # Store model and trainer
            self.models.append(model)
            self.trainers.append(trainer)
            histories.append(history)

            # Save individual model
            self._save_model(model, i, seed)

            print(f"\n✓ Model {i+1} complete - Best val loss: {trainer.best_val_loss:.4f}")

            # Free memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print("\n" + "="*80)
        print("ENSEMBLE TRAINING COMPLETE")
        print("="*80)

        # Save ensemble metadata
        self._save_ensemble_metadata(histories)

        return histories

    def _set_seed(self, seed):
        """Set random seed for reproducibility"""
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _save_model(self, model, index, seed):
        """Save individual model"""
        self.config.ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)

        path = self.config.ENSEMBLE_DIR / f'model_{index}_seed_{seed}.pth'
        torch.save(model.state_dict(), path)
        print(f"  Saved: {path}")

    def _save_ensemble_metadata(self, histories):
        """Save ensemble metadata and histories"""
        self.config.ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)

        metadata = {
            'ensemble_size': self.config.ENSEMBLE_SIZE,
            'seeds': self.config.ENSEMBLE_SEEDS[:self.config.ENSEMBLE_SIZE],
            'val_losses': [h['val_loss'][-1] for h in histories],
            'val_accs': [h['val_acc'][-1] for h in histories],
            'best_val_loss': min([h['val_loss'][-1] for h in histories]),
            'avg_val_acc': np.mean([h['val_acc'][-1] for h in histories]),
        }

        path = self.config.ENSEMBLE_DIR / 'ensemble_metadata.json'
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n✓ Saved ensemble metadata: {path}")

    def load_ensemble(self):
        """Load all models in ensemble"""
        print(f"\nLoading ensemble from {self.config.ENSEMBLE_DIR}...")

        self.models = []
        for i, seed in enumerate(self.config.ENSEMBLE_SEEDS[:self.config.ENSEMBLE_SIZE]):
            model = WeeklyPredictionModel(self.config)
            path = self.config.ENSEMBLE_DIR / f'model_{i}_seed_{seed}.pth'
            model.load_state_dict(torch.load(path, map_location=self.config.DEVICE))
            model.to(self.config.DEVICE)
            model.eval()
            self.models.append(model)

        print(f"✓ Loaded {len(self.models)} models")
        return self.models

    def ensemble_predict(self, x):
        """
        Make ensemble prediction by averaging all models.

        Args:
            x: Input features [batch, seq_len, features]

        Returns:
            predictions: dict of {asset: averaged_probabilities}
            attention_weights: averaged attention weights
        """
        if not self.models:
            raise ValueError("No models loaded. Call train_ensemble() or load_ensemble() first.")

        all_predictions = {asset: [] for asset in self.config.TARGET_ASSETS}
        all_attentions = []

        with torch.no_grad():
            for model in self.models:
                model.eval()
                preds, attn = model(x)

                # Collect predictions (apply sigmoid to get probabilities)
                for asset in self.config.TARGET_ASSETS:
                    prob = torch.sigmoid(preds[asset])
                    all_predictions[asset].append(prob)

                all_attentions.append(attn)

        # Average predictions across all models
        ensemble_predictions = {
            asset: torch.stack(all_predictions[asset]).mean(dim=0)
            for asset in self.config.TARGET_ASSETS
        }

        # Average attention weights
        ensemble_attention = torch.stack(all_attentions).mean(dim=0)

        return ensemble_predictions, ensemble_attention

    def evaluate_ensemble(self, test_loader):
        """
        Evaluate ensemble on test set.

        Args:
            test_loader: Test data loader

        Returns:
            metrics: Evaluation metrics
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

        print("\n" + "="*80)
        print("EVALUATING ENSEMBLE")
        print("="*80)

        all_preds = {asset: [] for asset in self.config.TARGET_ASSETS}
        all_labels = {asset: [] for asset in self.config.TARGET_ASSETS}

        with torch.no_grad():
            for features, labels in test_loader:
                features = features.to(self.config.DEVICE)

                # Ensemble prediction
                ensemble_preds, _ = self.ensemble_predict(features)

                # Collect
                for asset in self.config.TARGET_ASSETS:
                    pred_probs = ensemble_preds[asset].cpu().numpy().flatten()
                    true_labels = labels[asset].numpy().flatten()

                    all_preds[asset].extend(pred_probs)
                    all_labels[asset].extend(true_labels)

        # Compute metrics
        metrics = {}
        for asset in self.config.TARGET_ASSETS:
            y_true = np.array(all_labels[asset])
            y_pred_probs = np.array(all_preds[asset])
            y_pred_binary = (y_pred_probs > 0.5).astype(int)

            metrics[asset] = {
                'accuracy': accuracy_score(y_true, y_pred_binary),
                'precision': precision_score(y_true, y_pred_binary, zero_division=0),
                'recall': recall_score(y_true, y_pred_binary, zero_division=0),
                'f1': f1_score(y_true, y_pred_binary, zero_division=0),
                'roc_auc': roc_auc_score(y_true, y_pred_probs)
            }

            print(f"\n{asset}:")
            print(f"  Accuracy:  {metrics[asset]['accuracy']:.4f}")
            print(f"  Precision: {metrics[asset]['precision']:.4f}")
            print(f"  Recall:    {metrics[asset]['recall']:.4f}")
            print(f"  F1 Score:  {metrics[asset]['f1']:.4f}")
            print(f"  ROC-AUC:   {metrics[asset]['roc_auc']:.4f}")

        # Overall metrics
        metrics['overall'] = {
            'accuracy': np.mean([metrics[asset]['accuracy'] for asset in self.config.TARGET_ASSETS]),
            'precision': np.mean([metrics[asset]['precision'] for asset in self.config.TARGET_ASSETS]),
            'recall': np.mean([metrics[asset]['recall'] for asset in self.config.TARGET_ASSETS]),
            'f1': np.mean([metrics[asset]['f1'] for asset in self.config.TARGET_ASSETS]),
            'roc_auc': np.mean([metrics[asset]['roc_auc'] for asset in self.config.TARGET_ASSETS])
        }

        print(f"\nOVERALL ENSEMBLE:")
        print(f"  Accuracy:  {metrics['overall']['accuracy']:.4f}")
        print(f"  Precision: {metrics['overall']['precision']:.4f}")
        print(f"  Recall:    {metrics['overall']['recall']:.4f}")
        print(f"  F1 Score:  {metrics['overall']['f1']:.4f}")
        print(f"  ROC-AUC:   {metrics['overall']['roc_auc']:.4f}")
        print("="*80)

        return metrics

    def compare_with_individuals(self, test_loader):
        """Compare ensemble performance with individual models"""
        from sklearn.metrics import accuracy_score

        print("\n" + "="*80)
        print("ENSEMBLE VS INDIVIDUAL MODELS")
        print("="*80)

        individual_accs = []

        # Evaluate each individual model
        for i, model in enumerate(self.models):
            model.eval()
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for features, labels in test_loader:
                    features = features.to(self.config.DEVICE)
                    preds, _ = model(features)

                    for asset in self.config.TARGET_ASSETS:
                        prob = torch.sigmoid(preds[asset]).cpu().numpy().flatten()
                        pred_binary = (prob > 0.5).astype(int)
                        true_labels = labels[asset].numpy().flatten()

                        all_preds.extend(pred_binary)
                        all_labels.extend(true_labels)

            acc = accuracy_score(all_labels, all_preds)
            individual_accs.append(acc)
            print(f"  Model {i+1} (seed={self.config.ENSEMBLE_SEEDS[i]}): {acc:.4f}")

        # Ensemble accuracy
        ensemble_metrics = self.evaluate_ensemble(test_loader)
        ensemble_acc = ensemble_metrics['overall']['accuracy']

        print(f"\n  Individual Average: {np.mean(individual_accs):.4f}")
        print(f"  Ensemble:           {ensemble_acc:.4f}")
        print(f"  Improvement:        {ensemble_acc - np.mean(individual_accs):+.4f}")
        print("="*80)

        return {
            'individual_accs': individual_accs,
            'individual_avg': np.mean(individual_accs),
            'ensemble_acc': ensemble_acc,
            'improvement': ensemble_acc - np.mean(individual_accs)
        }
