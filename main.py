# main.py
"""
Main execution script for Multi-Asset Directional Prediction
"""
import torch
from torch.utils.data import DataLoader
import numpy as np
import random
from pathlib import Path

# Import project modules
from src.utils.config import Config
from src.data.loader import DataLoader as MultiAssetLoader
from src.data.features import FeatureEngineer
from src.data.dataset import MultiAssetDataset
from src.models.weekly_model import WeeklyPredictionModel  # NEW: Weekly/Monthly model
from src.training.trainer import Trainer
from src.evaluation.metrics import Evaluator
from src.evaluation.visualize import AttentionVisualizer
from src.evaluation.backtest import SimpleBacktest

def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    """Main execution pipeline"""

    # Initialize config
    config = Config()
    set_seed(config.RANDOM_SEED)

    print("\n" + "="*80)
    print(f"MULTI-ASSET {config.PREDICTION_HORIZON.upper()} PREDICTION WITH ATTENTION")
    print("="*80)
    print(f"Device: {config.DEVICE}")
    print(f"Random seed: {config.RANDOM_SEED}")
    print(f"Prediction horizon: {config.PREDICTION_HORIZONS[config.PREDICTION_HORIZON]} days ({config.PREDICTION_HORIZON})")
    print(f"Sequence length: {config.SEQUENCE_LENGTH} days")
    print("="*80 + "\n")

    # ====================================================================
    # STEP 1: LOAD AND PREPARE DATA
    # ====================================================================
    print("STEP 1: Loading and preparing data...")
    print("-" * 80)

    # Load raw data
    loader = MultiAssetLoader(config)
    df_raw = loader.get_data()

    # Feature engineering
    engineer = FeatureEngineer(config)
    df_features = engineer.engineer_features(df_raw)

    # Get feature columns
    feature_cols = engineer.get_feature_columns()
    print(f"Total features: {len(feature_cols)}")

    # Prepare labels dict
    labels = {}
    label_df = df_features.copy()
    for asset in config.TARGET_ASSETS:
        labels[asset] = label_df[f'{asset}_Label']

    # Split data chronologically
    n = len(df_features)
    train_end = int(n * config.TRAIN_SPLIT)
    val_end = int(n * (config.TRAIN_SPLIT + config.VAL_SPLIT))

    # Create datasets
    print("\nCreating datasets...")
    train_dataset = MultiAssetDataset(
        features=df_features[feature_cols].iloc[:train_end],
        labels={asset: df_features[f'{asset}_Label'].iloc[:train_end]
                for asset in config.TARGET_ASSETS},
        dates=df_features['Date'].iloc[:train_end],
        sequence_length=config.SEQUENCE_LENGTH,
        scaler=None,
        fit_scaler=True
    )

    val_dataset = MultiAssetDataset(
        features=df_features[feature_cols].iloc[train_end:val_end],
        labels={asset: df_features[f'{asset}_Label'].iloc[train_end:val_end]
                for asset in config.TARGET_ASSETS},
        dates=df_features['Date'].iloc[train_end:val_end],
        sequence_length=config.SEQUENCE_LENGTH,
        scaler=train_dataset.scaler,
        fit_scaler=False
    )

    test_dataset = MultiAssetDataset(
        features=df_features[feature_cols].iloc[val_end:],
        labels={asset: df_features[f'{asset}_Label'].iloc[val_end:]
                for asset in config.TARGET_ASSETS},
        dates=df_features['Date'].iloc[val_end:],
        sequence_length=config.SEQUENCE_LENGTH,
        scaler=train_dataset.scaler,
        fit_scaler=False
    )

    print(f"Train: {len(train_dataset)} sequences")
    print(f"Val:   {len(val_dataset)} sequences")
    print(f"Test:  {len(test_dataset)} sequences")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )

    # ====================================================================
    # STEP 2: INITIALIZE MODEL
    # ====================================================================
    print("\n" + "="*80)
    print("STEP 2: Initializing model...")
    print("-" * 80)

    model = WeeklyPredictionModel(config)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Model: Multi-Asset LSTM + Attention")
    print(f"Total parameters: {total_params:,}")
    print(f"LSTM hidden size: {config.LSTM_HIDDEN_SIZE}")
    print(f"LSTM layers: {config.LSTM_LAYERS}")
    print(f"Attention heads: {config.ATTENTION_HEADS}")

    # ====================================================================
    # STEP 3: TRAIN MODEL
    # ====================================================================
    print("\n" + "="*80)
    print("STEP 3: Training model...")
    print("="*80)

    trainer = Trainer(model, train_loader, val_loader, config)
    history = trainer.fit()

    # Load best model
    trainer.load_checkpoint()

    # Optimize thresholds on validation set
    trainer.optimize_thresholds()

    # ====================================================================
    # STEP 4: EVALUATE MODEL
    # ====================================================================
    print("\n" + "="*80)
    print("STEP 4: Evaluating model...")
    print("="*80)

    evaluator = Evaluator(model, test_loader, config)
    metrics, predictions, labels, probabilities = evaluator.evaluate()

    # Print and save metrics
    evaluator.print_metrics(metrics)
    metrics_df = evaluator.save_metrics(metrics)

    # ====================================================================
    # STEP 5: VISUALIZE RESULTS
    # ====================================================================
    print("\n" + "="*80)
    print("STEP 5: Creating visualizations...")
    print("="*80)

    visualizer = AttentionVisualizer(model, test_loader, config)

    # Plot training history
    print("\nPlotting training history...")
    visualizer.plot_training_history(history)

    # Plot attention heatmap
    print("\nPlotting attention heatmap...")
    attention = visualizer.plot_asset_attention_heatmap()

    # Analyze leading indicators
    influence_df = visualizer.analyze_leading_indicators(attention)

    # Plot confusion matrices
    print("\nPlotting confusion matrices...")
    visualizer.plot_confusion_matrices(metrics)

    # ====================================================================
    # STEP 6: SIMPLE BACKTEST
    # ====================================================================
    print("\n" + "="*80)
    print("STEP 6: Running simple backtest...")
    print("="*80)

    backtester = SimpleBacktest(model, test_dataset, config)
    backtest_results, backtest_metrics = backtester.run_backtest()

    # Print results
    backtester.print_backtest_results(backtest_metrics)

    # Plot cumulative returns
    print("\nPlotting backtest results...")
    backtester.plot_cumulative_returns(backtest_results)

    # ====================================================================
    # SUMMARY
    # ====================================================================
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)

    print("\nResults Summary:")
    print("-" * 80)
    print(f"Overall Accuracy:  {metrics['overall']['accuracy']:.3%}")
    print(f"Overall F1 Score:  {metrics['overall']['f1']:.3f}")
    print(f"Overall ROC-AUC:   {metrics['overall']['roc_auc']:.3f}")

    print("\nPer-Asset Performance:")
    for asset in config.TARGET_ASSETS:
        print(f"  {asset:12s}: Acc={metrics[asset]['accuracy']:.3%}, "
              f"F1={metrics[asset]['f1']:.3f}, "
              f"AUC={metrics[asset]['roc_auc']:.3f}")

    print("\nBacktest Performance (Phase 2 - Gated):")
    for asset in config.TARGET_ASSETS:
        p2 = backtest_metrics[asset]['phase2']
        print(f"  {asset:12s}: Return={p2['cumulative_return']:+.1f}, "
              f"Sharpe={p2['sharpe_ratio']:.2f}, "
              f"WinRate={p2['win_rate']:.1%}")

    print("\nSaved Files:")
    print(f"  Model:          {config.MODEL_DIR / 'best_model.pth'}")
    print(f"  Metrics:        {config.RESULTS_DIR / 'metrics.csv'}")
    print(f"  Visualizations: {config.PLOT_DIR / '*.png'}")

    print("\n" + "="*80)
    print("Check the results/ directory for all outputs!")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
