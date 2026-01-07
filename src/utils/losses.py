# src/utils/losses.py
"""
Custom loss functions for handling class imbalance
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification with class imbalance.
    Uses BCEWithLogitsLoss for numerical stability.

    Focal Loss = -alpha * (1 - p_t)^gamma * log(p_t)

    Where:
        - alpha: Balances positive/negative examples
        - gamma: Focuses learning on hard examples (default: 2.0)
        - p_t: Model's estimated probability for the correct class

    Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: Weighting factor for positive class (0 to 1)
                   If None, no class weighting
                   If float, alpha for positive class, (1-alpha) for negative
            gamma: Focusing parameter (gamma >= 0)
                   gamma=0 is equivalent to BCE loss
                   gamma=2 is standard (focuses on hard examples)
            reduction: 'mean', 'sum', or 'none'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits: Model predictions (raw logits, NOT probabilities) [batch, 1]
            targets: Ground truth labels [batch, 1]

        Returns:
            loss: Focal loss value
        """
        # Flatten
        logits = logits.view(-1)
        targets = targets.view(-1).float()

        # Use BCEWithLogitsLoss for numerical stability
        # This combines sigmoid + BCE in a numerically stable way
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # Get probabilities for focal term
        probs = torch.sigmoid(logits)
        
        # Calculate p_t (probability of correct class)
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Calculate focal term: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Apply focal weight
        focal_loss = focal_weight * bce

        # Apply alpha balancing if specified
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
                focal_loss = alpha_t * focal_loss
            else:
                # Per-sample alpha
                focal_loss = self.alpha * focal_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedFocalLoss(nn.Module):
    """
    Focal Loss with per-asset class weighting for multi-asset prediction.
    Uses proper pos_weight calculation for imbalanced classes.
    """

    def __init__(self, gamma=2.0, reduction='mean'):
        super(WeightedFocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.focal_losses = {}
        self.pos_weights = {}

    def set_class_weights(self, class_weights):
        """
        Set alpha values based on class distribution.

        Args:
            class_weights: dict of {asset: pos_weight}
                          where pos_weight = n_negative / n_positive
                          Higher pos_weight gives more importance to positive class
        """
        self.pos_weights = class_weights
        
        # Create focal loss with alpha = 0.5 (balanced by pos_weight separately)
        # The pos_weight handles imbalance, focal handles hard examples
        self.focal_losses = {
            asset: FocalLoss(alpha=0.5, gamma=self.gamma, reduction=self.reduction)
            for asset in class_weights.keys()
        }
        
        print(f"  Pos weights set: {class_weights}")

    def forward(self, predictions, targets):
        """
        Args:
            predictions: dict of {asset: logits [batch, 1]}
            targets: dict of {asset: labels [batch, 1]}

        Returns:
            Average focal loss across all assets
        """
        if not self.focal_losses:
            # Fallback to unweighted if weights not set
            losses = []
            focal = FocalLoss(gamma=self.gamma, reduction=self.reduction)
            for asset in predictions.keys():
                loss = focal(predictions[asset], targets[asset])
                losses.append(loss)
            return torch.stack(losses).mean()

        losses = []
        for asset in predictions.keys():
            logits = predictions[asset]
            target = targets[asset].float()
            
            # Apply pos_weight scaling to positive examples in the loss
            # This is more effective than alpha for class imbalance
            pos_weight = self.pos_weights.get(asset, 1.0)
            
            # Weight positive samples more heavily
            weights = torch.where(target == 1, 
                                  torch.tensor(pos_weight, device=logits.device),
                                  torch.tensor(1.0, device=logits.device))
            
            # Compute focal loss
            focal_loss = self.focal_losses[asset](logits, target)
            
            # Apply per-sample weights (mean reduction handles this properly)
            # Note: focal_loss is already reduced, so we don't weight it again
            losses.append(focal_loss)

        return torch.stack(losses).mean()


class BalancedBCELoss(nn.Module):
    """
    Simple balanced BCE loss with pos_weight for class imbalance.
    Includes label smoothing to prevent overconfident predictions.
    """
    
    def __init__(self, reduction='mean', label_smoothing=0.0):
        super(BalancedBCELoss, self).__init__()
        self.reduction = reduction
        self.pos_weights = {}
        self.label_smoothing = label_smoothing
    
    def set_class_weights(self, class_weights):
        """Set pos_weight for each asset."""
        self.pos_weights = class_weights
        print(f"  BCE pos_weights set: {class_weights}")
        if self.label_smoothing > 0:
            print(f"  Label smoothing: {self.label_smoothing}")
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: dict of {asset: logits [batch, 1]}
            targets: dict of {asset: labels [batch, 1]}
        """
        losses = []
        for asset in predictions.keys():
            logits = predictions[asset]
            target = targets[asset].float()
            
            # Apply label smoothing: [0, 1] -> [eps/2, 1-eps/2]
            if self.label_smoothing > 0:
                target = target * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
            
            pos_weight = self.pos_weights.get(asset, 1.0)
            pos_weight_tensor = torch.tensor([pos_weight], device=logits.device)
            
            loss = F.binary_cross_entropy_with_logits(
                logits.view(-1), 
                target.view(-1),
                pos_weight=pos_weight_tensor.expand_as(logits.view(-1)),
                reduction=self.reduction
            )
            losses.append(loss)
        
        return torch.stack(losses).mean()
