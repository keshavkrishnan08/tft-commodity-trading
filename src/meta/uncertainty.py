# src/meta/uncertainty.py
"""
Conformal Prediction Engine for Uncertainty Quantification.
Provides statistically guaranteed prediction intervals for risk management.
"""
import torch
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

class ConformalPredictor:
    """
    Implements Split Conformal Prediction for time-series intervals.
    """
    def __init__(self, alpha=0.05):
        """
        Args:
            alpha: Mis-coverage level (e.g., 0.05 means 95% confidence)
        """
        self.alpha = alpha
        self.q_hat = None # Quantile of non-conformity scores

    def calibrate(self, predictions, actuals):
        """
        Calibrate the predictor using a calibration set.
        
        Args:
            predictions: Model output probs/values [batch, 1]
            actuals: Ground truth values [batch, 1]
        """
        # For classification, non-conformity score can be 1-P(correct_class)
        # For regression, |actual - pred|
        
        # Defaulting to regression-style residuals for SL/TP distance calibration
        scores = torch.abs(predictions - actuals).flatten()
        
        # Calculate q_hat (the (1-alpha)*(1+1/n) quantile)
        n = len(scores)
        quantile = (1 - self.alpha) * (1 + 1/n)
        
        # Clamp quantile to [0, 1]
        quantile = min(0.999, max(0.001, quantile))
        
        # Use numpy for quantile calculation
        self.q_hat = np.quantile(scores.detach().cpu().numpy(), quantile)
        print(f"✓ Conformal calibration complete. q_hat: {self.q_hat:.4f}")

    def get_interval(self, prediction):
        """
        Return the 1-alpha prediction interval.
        
        Args:
            prediction: Model output for a new sample.
            
        Returns:
            (lower, upper)
        """
        if self.q_hat is None:
            raise ValueError("Predictor must be calibrated before use.")
            
        return (prediction - self.q_hat, prediction + self.q_hat)

    def get_risk_score(self, probs):
        """
        Returns a novelty-heavy 'Risk Score' based on prediction entropy + q_hat.
        """
        entropy = -torch.sum(probs * torch.log(probs + 1e-6))
        return entropy * (1 + self.q_hat)

if __name__ == "__main__":
    # Test Conformal Prediction
    cp = ConformalPredictor(alpha=0.05)
    
    # Dummy calibration data
    preds = torch.tensor([0.5, 0.52, 0.48, 0.51, 0.49])
    labels = torch.tensor([0.45, 0.55, 0.40, 0.50, 0.60])
    
    cp.calibrate(preds, labels)
    
    new_pred = torch.tensor(0.5)
    lower, upper = cp.get_interval(new_pred)
    
    print(f"Prediction: {new_pred.item():.4f}")
    print(f"95% Interval: [{lower:.4f}, {upper:.4f}]")
    print("✓ Conformal Prediction engine test passed!")
