import torch
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
from src.utils.config import Config
from src.data.loader import DataLoader as MultiAssetLoader
from src.data.calibrated_features import CalibratedFeatureEngineer
from src.data.tft_dataset import TFTDataset
from src.models.temporal_fusion_transformer import TemporalFusionTransformer

def analyze_vsn_entropy():
    print("="*80)
    print("ANALYZING VSN ENTROPY (NATURE MI EVIDENCE)")
    print("="*80)
    
    config = Config()
    config.TFT_HIDDEN_DIM = 32
    config.TFT_NUM_HEADS = 4
    config.TFT_NUM_LAYERS = 2
    config.TFT_DROPOUT = 0.5
    
    EXPERIMENT_DIR = Path('experiments/tft_v8_expanded')
    MODEL_DIR = EXPERIMENT_DIR / 'models'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Data
    loader = MultiAssetLoader(config)
    df_raw = loader.get_data()
    engineer = CalibratedFeatureEngineer(config, d=0.4)
    df = engineer.engineer_features(df_raw)
    
    exclude_cols = ['Date'] + [c for c in df.columns if 'Label' in c or 'FutureReturn' in c]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    config.INPUT_DIM = len(feature_cols)
    
    # 2. Load Model
    model = TemporalFusionTransformer(config).to(device)
    model.load_state_dict(torch.load(MODEL_DIR / 'tft_best.pt', map_location=device))
    model.eval()
    scaler = joblib.load(MODEL_DIR / 'tft_scaler.pkl')
    
    # 3. Create Dataset (Full range for analysis)
    dataset = TFTDataset(
        features=df[feature_cols],
        labels={a: df[f'{a}_Label'] for a in config.TARGET_ASSETS},
        dates=df['Date'],
        sequence_length=config.SEQUENCE_LENGTH,
        scaler=scaler,
        fit_scaler=False
    )
    
    # 4. Extract VSN Weights
    print("Extracting VSN weights...")
    results = []
    
    # Define focus period: 2021-2022 (Crypto Crash & Inflation)
    # We'll just run inference on the whole set and filter later
    
    dates = []
    entropies = {a: [] for a in config.TARGET_ASSETS}
    
    with torch.no_grad():
        for i in range(len(dataset)):
            if i % 500 == 0:
                print(f"  Processed {i}/{len(dataset)} steps")
                
            features, time_feats, _ = dataset[i]
            x = features.unsqueeze(0).to(device)
            t = time_feats.unsqueeze(0).to(device)
            
            # Forward pass to get VSN weights
            # We don't need time_feats for VSN, just x
            vsn_weights = model.get_vsn_weights(x)  # (1, seq, num_features)
            
            # Take the average weight over the sequence (or last step?)
            # Let's verify shape: (1, seq, num_features)
            # We want entropy at the decision point (last step) or average?
            # Start with last step for "current regime"
            w = vsn_weights[0, -1, :].cpu().numpy() # (num_features,)
            # Normalize just in case
            w = w / (w.sum() + 1e-9)
            entropy = -np.sum(w * np.log(w + 1e-9))
            
            date = dataset.get_date(i)
            dates.append(date)
            
            # Store same entropy for all assets (shared backbone) 
            # OR if we have per-asset heads, check that. 
            # TFT backbone is shared.
            for asset in config.TARGET_ASSETS:
                entropies[asset].append(entropy)

    # 5. Save Results
    df_entropy = pd.DataFrame({'Date': dates})
    for asset in config.TARGET_ASSETS:
        df_entropy[f'{asset}_Entropy'] = entropies[asset]
        
    out_file = EXPERIMENT_DIR / 'vsn_entropy.csv'
    df_entropy.to_csv(out_file, index=False)
    print(f"Entropy data saved to {out_file}")
    
    # 6. Plotting specific regimes
    print("Generating Entropy vs Price plots...")
    
    # Plot BTC during 2021-2022 Crash
    btc_df = df_raw[['Date', 'BTC_Close']].copy()
    btc_df = btc_df.merge(df_entropy[['Date', 'BTC_Entropy']], on='Date')
    btc_df = btc_df[(btc_df['Date'] >= '2021-01-01') & (btc_df['Date'] <= '2022-12-31')]
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(btc_df['Date'], btc_df['BTC_Close'], color='orange', label='BTC Price')
    ax1.set_ylabel('Price', color='orange')
    
    ax2 = ax1.twinx()
    ax2.plot(btc_df['Date'], btc_df['BTC_Entropy'], color='purple', alpha=0.6, label='VSN Entropy')
    ax2.set_ylabel('VSN Entropy (Uncertainty)', color='purple')
    
    plt.title('BTC Price vs VSN Entropy (2021-2022)')
    plt.savefig(EXPERIMENT_DIR / 'btc_entropy_crash.png')
    
    print("Plots saved.")

if __name__ == "__main__":
    analyze_vsn_entropy()
