# src/utils/config.py
"""
Configuration for Multi-Asset Directional Prediction Model
"""
import torch
from pathlib import Path

class Config:
    """Project configuration"""

    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_DIR = PROJECT_ROOT / 'data'
    RESULTS_DIR = PROJECT_ROOT / 'experiments' / 'winrate_optimization_v3'
    MODEL_DIR = RESULTS_DIR / 'models'
    PLOT_DIR = RESULTS_DIR / 'plots'

    # Data files
    OIL_GAS_FILE = DATA_DIR / 'oil and gas.csv'
    METALS_CRYPTO_FILE = DATA_DIR / 'metals_crypto.csv'
    DXY_FILE = DATA_DIR / 'dxy.csv'

    # Assets to predict (targets)
    TARGET_ASSETS = ['WTI', 'Brent', 'NaturalGas', 'HeatingOil', 'Gold', 'Silver', 'BTC']

    # All assets (including DXY as confluence factor)
    ALL_ASSETS = TARGET_ASSETS + ['DXY']

    # Asset name mapping from CSV
    ASSET_NAME_MAP = {
        'Crude Oil WTI': 'WTI',
        'Brent Oil': 'Brent',
        'Natural Gas': 'NaturalGas',
        'Heating Oil': 'HeatingOil',
        'Gold': 'Gold',
        'Silver': 'Silver',
        'BTC': 'BTC',
        'US Dollar Index': 'DXY'
    }

    # Data parameters
    START_DATE = '2000-01-01'
    END_DATE = '2022-12-31'
    TRAIN_SPLIT = 0.70
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15

    # Feature engineering
    SEQUENCE_LENGTH = 90  # 90-day lookback (increased for longer-term patterns)

    # Technical indicators
    VOLATILITY_WINDOWS = [5, 10, 20, 60]  # Rolling volatility windows
    SMA_WINDOWS = [10, 20, 50, 200]       # Simple moving averages (added long-term)
    EMA_WINDOWS = [12, 26, 50]            # Exponential moving averages
    RSI_PERIOD = 14                       # RSI period
    MACD_FAST = 12                        # MACD fast period
    MACD_SLOW = 26                        # MACD slow period
    MACD_SIGNAL = 9                       # MACD signal period

    # Prediction horizons (NEW!)
    PREDICTION_HORIZONS = {
        'weekly': 5,      # 5 trading days = 1 week
        'biweekly': 10,   # 10 trading days = 2 weeks
        'monthly': 21     # 21 trading days ≈ 1 month
    }

    # Use which horizon for training
    PREDICTION_HORIZON = 'weekly'  # Options: 'weekly', 'biweekly', 'monthly'

    # Directional labeling thresholds (higher for longer timeframes)
    MOVE_THRESHOLDS = {
        'weekly': 0.02,     # 2% move for weekly
        'biweekly': 0.03,   # 3% move for biweekly
        'monthly': 0.05     # 5% move for monthly
    }

    # Model architecture
    LSTM_HIDDEN_SIZE = 128
    LSTM_LAYERS = 2
    ATTENTION_HEADS = 8
    DROPOUT = 0.5  # Increased from 0.3 to reduce overfitting

    # Training
    BATCH_SIZE = 64
    LEARNING_RATE = 5e-4  # Reduced from 1e-3 to prevent premature convergence
    EPOCHS = 5  # Rapid Sniper Validation
    EARLY_STOP_PATIENCE = 15
    LR_SCHEDULE_PATIENCE = 5
    LR_SCHEDULE_FACTOR = 0.5

    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Random seed
    RANDOM_SEED = 42

    # Backtest
    # VERY AGGRESSIVE: Take trades on any conviction above 50.5/49.5
    # This maximizes trading frequency and minimizes gaps
    BUY_THRESHOLD = 0.505
    SELL_THRESHOLD = 0.495
    MIN_SHARPE_THRESHOLD = 1.0  # Novel: Only trade if strategy Sharpe > 1
    ROLLING_SHARPE_WINDOW = 20  # Window for calculating rolling Sharpe
    
    # Capital and Risk Management
    INITIAL_CAPITAL = 10000  # Starting capital in dollars
    MAX_POSITION_SIZE = 0.10  # Maximum 10% of capital per trade
    MIN_POSITION_SIZE = 0.01  # Minimum 1% of capital per trade
    DEFAULT_STOP_LOSS = 0.02  # Default 2% stop loss
    DEFAULT_TAKE_PROFIT = 0.04  # Default 4% take profit
    
    # Advanced Risk Management (v2)
    SLIPPAGE_PCT = 0.0002     # 0.02% slippage per trade
    COMMISSION_PCT = 0.0001   # 0.01% commission per trade
    ENABLE_TRAILING_STOP = True
    TRAILING_STOP_ATR_MULT = 2.0
    TRAILING_STOP_ACTIVATION_RR = 1.5
    
    # Label Smoothing (prevents overconfident predictions)
    LABEL_SMOOTHING = 0.1  # Smooth labels from [0,1] to [0.05, 0.95]

    # =========================================================================
    # ENSEMBLE + ADAMW + COSINE ANNEALING SETTINGS
    # =========================================================================
    
    # Ensemble settings
    ENSEMBLE_SIZE = 5
    ENSEMBLE_SEEDS = [42, 123, 456, 789, 999]
    ENSEMBLE_DIR = RESULTS_DIR / 'ensemble'

    # AdamW settings
    WEIGHT_DECAY = 1e-4  # Decoupled weight decay for AdamW

    # Cosine Annealing settings
    COSINE_T0 = 10        # Initial restart period (epochs)
    COSINE_T_MULT = 2     # Period multiplier after each restart
    COSINE_ETA_MIN = 1e-6 # Minimum learning rate
