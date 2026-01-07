import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from pathlib import Path

def plot_reliability_diagram(y_true, y_prob, asset_name, save_path):
    """
    Generates a reliability diagram to show calibration quality.
    Perfectly calibrated models follow the 45-degree diagonal.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
    plt.plot(prob_pred, prob_true, marker='o', label=f'{asset_name} (Raw)')
    
    plt.xlabel('Predicted Probability')
    plt.ylabel('Actual Win Frequency')
    plt.title(f'Reliability Diagram: {asset_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(save_path)
    plt.close()
    print(f"✓ Saved reliability diagram to {save_path}")

if __name__ == "__main__":
    # Example diagnostic run
    dummy_true = np.random.randint(0, 2, 1000)
    dummy_prob = np.clip(np.random.normal(0.6, 0.1, 1000), 0, 1) # Skewed high
    
    Path("results/plots/diagnostics").mkdir(parents=True, exist_ok=True)
    plot_reliability_diagram(dummy_true, dummy_prob, "WTI Sample", "results/plots/diagnostics/wti_calibration.png")
