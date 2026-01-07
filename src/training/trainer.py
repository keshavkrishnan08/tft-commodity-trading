# src/training/trainer.py
"""
Training pipeline for multi-asset directional prediction
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import Config
from src.utils.losses import WeightedFocalLoss

class Trainer:
    """Training pipeline with early stopping and LR scheduling"""

    def __init__(self, model, train_loader, val_loader, config=None):
        self.config = config if config else Config()
        self.model = model.to(self.config.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Calculate class weights from training data
        print("\nCalculating class weights from training data...")
        class_weights = self._calculate_class_weights()

        # Loss function: Balanced BCE Loss with label smoothing
        # Label smoothing prevents overconfident predictions (always up/down)
        from src.utils.losses import BalancedBCELoss
        self.criterion = BalancedBCELoss(
            reduction='mean', 
            label_smoothing=self.config.LABEL_SMOOTHING
        )
        self.criterion.set_class_weights(class_weights)
        print("✓ Using Balanced BCE Loss with pos_weights + label smoothing")

        # Optimizer (AdamW with decoupled weight decay)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # Learning rate scheduler (Cosine Annealing with Warm Restarts)
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config.COSINE_T0,
            T_mult=self.config.COSINE_T_MULT,
            eta_min=self.config.COSINE_ETA_MIN
        )

        # Early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # Asset-specific thresholds (optimized on validation set)
        self.asset_thresholds = {asset: 0.5 for asset in self.config.TARGET_ASSETS}

        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'lr': []
        }

    def _calculate_class_weights(self):
        """Calculate class weights for each asset from training data"""
        class_counts = {asset: {'pos': 0, 'neg': 0} for asset in self.config.TARGET_ASSETS}

        # Count positive and negative samples for each asset
        for _, labels in self.train_loader:
            for asset in self.config.TARGET_ASSETS:
                labels_asset = labels[asset].numpy().flatten()
                class_counts[asset]['pos'] += (labels_asset == 1).sum()
                class_counts[asset]['neg'] += (labels_asset == 0).sum()

        # Calculate alpha (pos_weight for positive class)
        # pos_weight = n_negative / n_positive
        # This gives more weight to under-represented positive class
        class_weights = {}
        print("\nClass distribution and pos_weights:")
        for asset in self.config.TARGET_ASSETS:
            n_pos = class_counts[asset]['pos']
            n_neg = class_counts[asset]['neg']
            
            # pos_weight = n_neg / n_pos (weight for positive class)
            # If balanced (50/50): pos_weight = 1.0
            # If more negatives: pos_weight > 1.0 (upweight positives)
            # If more positives: pos_weight < 1.0 (downweight positives)
            pos_weight = n_neg / (n_pos + 1e-6)
            
            # Clamp to reasonable range to avoid extreme weights
            pos_weight = max(0.5, min(2.0, pos_weight))
            class_weights[asset] = pos_weight

            print(f"  {asset}: pos={n_pos}, neg={n_neg}, "
                  f"ratio={n_pos/(n_pos+n_neg):.3f}, pos_weight={pos_weight:.3f}")

        return class_weights

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = {asset: 0 for asset in self.config.TARGET_ASSETS}
        total = 0

        for features, labels in self.train_loader:
            features = features.to(self.config.DEVICE)
            labels = {asset: labels[asset].to(self.config.DEVICE)
                     for asset in self.config.TARGET_ASSETS}

            # Forward pass
            self.optimizer.zero_grad()
            predictions, attention_weights = self.model(features)

            # Compute loss for each asset (focal loss with class weights)
            loss = self.criterion(predictions, labels)
            batch_size = features.size(0)

            # Accuracy computation
            for asset in self.config.TARGET_ASSETS:
                pred = predictions[asset]
                target = labels[asset]
                pred_class = (pred > 0.5).float().view(-1)
                target = target.float().view(-1)
                correct[asset] += (pred_class == target).sum().item()

            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients in LSTM
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()

            total_loss += loss.item()
            total += batch_size

        # Average metrics
        avg_loss = total_loss / len(self.train_loader)
        avg_acc = np.mean([correct[asset] / total for asset in self.config.TARGET_ASSETS])

        return avg_loss, avg_acc

    def validate(self):
        """Validate on validation set"""
        self.model.eval()
        total_loss = 0
        correct = {asset: 0 for asset in self.config.TARGET_ASSETS}
        total = 0

        with torch.no_grad():
            for features, labels in self.val_loader:
                features = features.to(self.config.DEVICE)
                labels = {asset: labels[asset].to(self.config.DEVICE)
                         for asset in self.config.TARGET_ASSETS}

                # Forward pass (no entropy in validation)
                predictions, _ = self.model(features)

                # Compute loss using focal loss (pass dicts)
                loss = self.criterion(predictions, labels)
                batch_size = features.size(0)

                # Accuracy computation
                for asset in self.config.TARGET_ASSETS:
                    pred = predictions[asset]
                    target = labels[asset]
                    pred_class = (pred > 0.5).float().view(-1)
                    target = target.float().view(-1)
                    correct[asset] += (pred_class == target).sum().item()

                total_loss += loss.item()
                total += batch_size

        avg_loss = total_loss / len(self.val_loader)
        avg_acc = np.mean([correct[asset] / total for asset in self.config.TARGET_ASSETS])

        return avg_loss, avg_acc

    def fit(self, epochs=None):
        """Train the model"""
        if epochs is None:
            epochs = self.config.EPOCHS

        print("\n" + "="*80)
        print(f"TRAINING ON {self.config.DEVICE.upper()}")
        print("="*80)
        print(f"Epochs: {epochs}")
        print(f"Batch size: {self.config.BATCH_SIZE}")
        print(f"Learning rate: {self.config.LEARNING_RATE}")
        print(f"Early stopping patience: {self.config.EARLY_STOP_PATIENCE}")
        print("="*80 + "\n")

        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch()

            # Validate
            val_loss, val_acc = self.validate()

            # Learning rate scheduling (Cosine Annealing steps each epoch)
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)

            # Print progress
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Train Acc: {train_acc:.3f} | "
                  f"Val Acc: {val_acc:.3f} | "
                  f"LR: {current_lr:.6f}")

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_loss)
                print(f"  ✓ Best model saved (val_loss: {val_loss:.4f})")
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.config.EARLY_STOP_PATIENCE:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break

        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print("="*80 + "\n")

        return self.history

    def save_checkpoint(self, epoch, val_loss):
        """Save model checkpoint"""
        self.config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'history': self.history
        }

        path = self.config.MODEL_DIR / 'best_model.pth'
        torch.save(checkpoint, path)

    def load_checkpoint(self, path=None):
        """Load model checkpoint"""
        if path is None:
            path = self.config.MODEL_DIR / 'best_model.pth'

        checkpoint = torch.load(path, map_location=self.config.DEVICE)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']

        print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']+1}")
        print(f"  Val loss: {checkpoint['val_loss']:.4f}")

        return checkpoint

    def load_best_model(self):
        """Alias for load_checkpoint() which defaults to best_model.pth"""
        return self.load_checkpoint()

    def optimize_thresholds(self):
        """
        Optimize classification thresholds for each asset on validation set.
        Instead of using 0.5, find the threshold that maximizes F1 score.
        """
        from sklearn.metrics import roc_curve, f1_score

        print("\n" + "="*80)
        print("OPTIMIZING ASSET-SPECIFIC THRESHOLDS")
        print("="*80)

        self.model.eval()

        # Collect all predictions and labels for each asset
        all_preds = {asset: [] for asset in self.config.TARGET_ASSETS}
        all_labels = {asset: [] for asset in self.config.TARGET_ASSETS}

        with torch.no_grad():
            for features, labels in self.val_loader:
                features = features.to(self.config.DEVICE)

                predictions, _ = self.model(features)

                for asset in self.config.TARGET_ASSETS:
                    pred = predictions[asset].cpu().numpy().flatten()
                    label = labels[asset].numpy().flatten()

                    all_preds[asset].extend(pred)
                    all_labels[asset].extend(label)

        # Find optimal threshold for each asset
        for asset in self.config.TARGET_ASSETS:
            y_true = np.array(all_labels[asset])
            y_pred = np.array(all_preds[asset])

            # Compute ROC curve
            fpr, tpr, thresholds = roc_curve(y_true, y_pred)

            # Find threshold that maximizes Youden's J statistic (TPR - FPR)
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            optimal_threshold = thresholds[optimal_idx]

            # Also compute F1 at this threshold
            y_pred_binary = (y_pred >= optimal_threshold).astype(int)
            f1 = f1_score(y_true, y_pred_binary)

            self.asset_thresholds[asset] = optimal_threshold

            print(f"{asset}:")
            print(f"  Optimal threshold: {optimal_threshold:.4f} (default: 0.5000)")
            print(f"  F1 score at optimal: {f1:.4f}")
            print(f"  Sensitivity (TPR): {tpr[optimal_idx]:.4f}")
            print(f"  Specificity (1-FPR): {1-fpr[optimal_idx]:.4f}")

        print("="*80)
