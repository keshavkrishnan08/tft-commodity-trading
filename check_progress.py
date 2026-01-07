# check_progress.py
"""
Check training progress in real-time
"""
import time
from pathlib import Path

def check_progress():
    """Monitor training progress"""

    results_dir = Path('results')
    model_path = results_dir / 'models' / 'best_model.pth'
    metrics_path = results_dir / 'metrics.csv'

    print("\n" + "="*80)
    print("TRAINING PROGRESS CHECK")
    print("="*80)

    # Check if model exists
    if model_path.exists():
        size_mb = model_path.stat().st_size / 1024 / 1024
        mod_time = time.ctime(model_path.stat().st_mtime)
        print(f"\n✓ Model checkpoint found:")
        print(f"  Path: {model_path}")
        print(f"  Size: {size_mb:.2f} MB")
        print(f"  Last updated: {mod_time}")
    else:
        print("\n⏳ Model checkpoint not yet created (training in progress...)")

    # Check if metrics exist
    if metrics_path.exists():
        import pandas as pd
        df = pd.read_csv(metrics_path)
        print(f"\n✓ Evaluation complete!")
        print("\nResults:")
        print(df.to_string(index=False))
    else:
        print("\n⏳ Evaluation not yet complete")

    # Check for plots
    plot_dir = results_dir / 'plots'
    if plot_dir.exists():
        plots = list(plot_dir.glob('*.png'))
        if plots:
            print(f"\n✓ {len(plots)} visualization(s) generated:")
            for plot in plots:
                print(f"  - {plot.name}")
        else:
            print("\n⏳ Visualizations not yet generated")
    else:
        print("\n⏳ Plot directory not yet created")

    print("\n" + "="*80)
    print("TIP: Run this script again in a few minutes to check progress")
    print("="*80 + "\n")

if __name__ == "__main__":
    check_progress()
