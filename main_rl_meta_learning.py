
# main_rl_meta_learning.py
"""
Backtest using Reinforced Learning Meta-Learner.
Uses PPO Agent to make trading decisions.
"""
import torch
from torch.utils.data import DataLoader
import numpy as np
import random
from pathlib import Path
import pandas as pd

from src.utils.config import Config
from src.data.loader import DataLoader as MultiAssetLoader
from src.data.features import FeatureEngineer
from src.data.dataset import MultiAssetDataset
from src.models.weekly_model import WeeklyPredictionModel
from src.models.rl_meta_learner import RLMetaLearner
from src.evaluation.advanced_backtest import AdvancedBacktest
from src.visualization.trade_analytics import TradeAnalytics, create_summary_dashboard

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class RLBacktest(AdvancedBacktest):
    """Backtest using RL Agent"""
    def __init__(self, config, meta_learner):
        super().__init__(config)
        self.meta_learner = meta_learner

    def run_backtest(self, predictions, labels, dates, dataset):
        n_samples = len(dates)
        self.capital = 10000.0
        self.equity_curve = [self.capital]
        self.trades = []
        self.daily_returns = []
        self.max_equity = self.capital
        self.max_drawdown = 0.0
        
        # RL Agent doesn't have "online learning" enabled during test properly yet
        # But maintains state if needed.
        
        print(f"\nRunning RL BACKTEST on {n_samples} samples...")
        
        for i in range(n_samples):
            date = dates[i]
            daily_pnl = 0.0
            
            for asset in self.config.TARGET_ASSETS:
                raw_prob = predictions[asset][i]
                actual = labels[asset][i]
                
                # Market State extraction (simplified for backtest loop efficiency)
                # Ideally should match exactly training extraction
                volatility = self._estimate_volatility(dataset, asset, i)
                
                try:
                    feature_idx = i + dataset.sequence_length - 1
                    raw_features = dataset.features.iloc[feature_idx]
                    
                    market_state = {
                        'volatility': volatility,
                        'momentum_20d': raw_features.get(f'{asset}_Return', 0.0) * 20,
                        'momentum_60d': raw_features.get(f'{asset}_Return', 0.0) * 60,
                        'rsi': raw_features.get(f'{asset}_RSI', 50.0), # RAW value for RL
                        'macd_hist': raw_features.get(f'{asset}_MACD_Hist', 0.0),
                        'volume_surge': 1.0,
                    }
                except:
                    market_state = {'volatility': 0.02, 'momentum_20d': 0, 'momentum_60d': 0, 'rsi': 50, 'macd_hist': 0}

                # Account State
                current_drawdown = (self.max_equity - self.capital) / self.max_equity if self.max_equity > 0 else 0
                account_state = {
                    'capital': self.capital,
                    'drawdown': current_drawdown,
                    'recent_win_rate': 0.5, # RL agent might not use this dynamically yet
                    'recent_sharpe': 0.0,
                    'consecutive_wins': 0,
                    'consecutive_losses': 0
                }
                
                # === AGENT DECISION ===
                decision = self.meta_learner.predict_decision(raw_prob, market_state, account_state)
                
                if decision['should_take']:
                    direction = 'long' if raw_prob > 0.5 else 'short'
                    position_fraction = decision['position_size']
                    position_dollars = self.capital * position_fraction
                    
                    stop_loss_pct = max(0.012, volatility * 1.5)
                    take_profit_pct = stop_loss_pct * 2.0
                    
                    trade_decision = {
                        'take_trade': True,
                        'direction': direction,
                        'raw_probability': raw_prob,
                        'calibrated_probability': raw_prob,
                        'confidence': decision['confidence'],
                        'position_fraction': position_fraction,
                        'position_dollars': position_dollars,
                        'stop_loss_pct': stop_loss_pct,
                        'take_profit_pct': take_profit_pct,
                        'risk_reward': 2.0,
                        'volatility': volatility
                    }
                    
                    result = self._simulate_trade(trade_decision, actual, asset, date, volatility, i, dataset)
                    
                    if result:
                        daily_pnl += result['pnl']
                        self.trades.append(result)
            
            self.capital += daily_pnl
            self.equity_curve.append(self.capital)
            if self.capital > self.max_equity: 
                self.max_equity = self.capital
            dd = (self.max_equity - self.capital) / self.max_equity
            self.max_drawdown = max(self.max_drawdown, dd)
            
        return self._generate_results(dates)


def main():
    config = Config()
    # OVERRIDE OUTPUT DIR
    config.PLOT_DIR = Path('experiments/rl_sharpe_optimized_v2/plots')
    config.PLOT_DIR.mkdir(parents=True, exist_ok=True)
    
    set_seed(42)
    print("RL META-LEARNER BACKTEST")
    
    # 1. Load Direction Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    direction_model = WeeklyPredictionModel(config).to(device)
    try:
        direction_model.load_state_dict(torch.load('experiments/validated_experiment_v1/models/best_model.pth', map_location=device)['model_state_dict'])
        direction_model.eval()
    except:
        print("Failed to load Direction Model!")
        return

    # 2. Load RL Agent
    rl_agent = RLMetaLearner(config)
    try:
        rl_agent.load('experiments/rl_sharpe_optimized_v2/models/rl_meta_agent.pth')
        print("Loaded Trained RL Agent.")
    except:
        print("Failed to load RL Agent! Run train_rl_meta.py first.")
        return

    # 3. Load Data
    loader = MultiAssetLoader(config)
    df_raw = loader.get_data()
    engineer = FeatureEngineer(config)
    df = engineer.engineer_features(df_raw)
    
    # 4. Create Test Set
    feature_cols = engineer.get_feature_columns()
    n = len(df)
    train_end = int(n * config.TRAIN_SPLIT)
    val_end = int(n * (config.TRAIN_SPLIT + config.VAL_SPLIT))
    
    def extract_raw_prices(df):
        raw_prices = {}
        for asset in config.TARGET_ASSETS:
            cols = [f'{asset}_{c}' for c in ['Open', 'High', 'Low', 'Close']]
            raw_prices[asset] = df[cols]
        return raw_prices

    test_dataset = MultiAssetDataset(
        features=df[feature_cols].iloc[val_end:],
        labels={asset: df[f'{asset}_Label'].iloc[val_end:] for asset in config.TARGET_ASSETS},
        dates=df['Date'].iloc[val_end:],
        sequence_length=config.SEQUENCE_LENGTH,
        raw_prices=extract_raw_prices(df.iloc[val_end:])
    )
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 5. Generate Predictions (Env State)
    print("Generating base predictions...")
    all_predictions = {asset: [] for asset in config.TARGET_ASSETS}
    all_labels = {asset: [] for asset in config.TARGET_ASSETS}
    all_dates = []
    
    with torch.no_grad():
        for i, (features, labels) in enumerate(test_loader):
            features = features.to(device)
            preds, _ = direction_model(features)
            
            for asset in config.TARGET_ASSETS:
                probs = torch.sigmoid(preds[asset]).cpu().numpy().flatten()
                all_predictions[asset].extend(probs)
                all_labels[asset].extend(labels[asset].numpy().flatten())
            
            for j in range(len(features)):
                idx = i * 32 + j
                if idx < len(test_dataset):
                    all_dates.append(test_dataset.get_date(idx))

    # 6. Run RL Backtest
    backtester = RLBacktest(config, rl_agent)
    results = backtester.run_backtest(all_predictions, all_labels, all_dates, test_dataset)
    
    # 7. Analysis
    backtester.print_results(results)
    analytics = TradeAnalytics(results, save_dir=config.PLOT_DIR)
    analytics.plot_all()
    create_summary_dashboard(results, config.PLOT_DIR / 'rl_dashboard.png')
    
    # Save Metrics
    metrics = pd.DataFrame([results])
    metrics.to_csv('experiments/rl_sharpe_optimized_v2/metrics.csv', index=False)
    print("Results saved to experiments/rl_sharpe_optimized_v2/")

if __name__ == "__main__":
    main()
