import pandas as pd
import os
import sys

def prepare_ppo_data(model_name="gemini", ticker=None):
    """
    Prepares PPO-ready data by copying the reference NLP feature CSV.
    
    The actual OHLCV + technical indicator data is now downloaded via yfinance 
    directly in train_ppo.py (matching the reference notebook pipeline).
    This function only handles the NLP feature CSV preparation.
    
    Column naming convention:
    - The raw CSV files use Title Case names (e.g., 'News Relevance', 'Price Impact Potential')
    - train_ppo.py renames them to snake_case (e.g., 'news_relevance', 'price_impact_potential')
      to match the reference config.py FUNDAMENTAL_INDICATORS
    """
    # Map model names to their source data directories
    # Reference data is stored in the PrimoGPT-main notebooks directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Source: reference NLP feature data
    ref_data_dir = os.path.join(base_dir, '..', 'PrimoGPT-main', 'notebooks', '6. PrimoRL trading with NLP features', 'data')
    
    tickers = [ticker] if ticker else ["AAPL", "AMZN", "CRM", "MSFT", "NFLX"]
    
    for t in tickers:
        src_path = os.path.join(ref_data_dir, f'{t}_data.csv')
        
        if not os.path.exists(src_path):
            print(f"Cannot find reference data at: {src_path}")
            continue
        
        print(f"Reading NLP feature data from: {src_path}")
        df = pd.read_csv(src_path)
        
        # Save formatted PPO-ready data
        out_dir = os.path.join(base_dir, 'ppo_data')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'{model_name}_ppo_{t}_data.csv')
        df.to_csv(out_path, index=False)
        print(f"Successfully generated PPO formatted data for '{model_name}' / '{t}': {out_path}")

if __name__ == '__main__':
    # Generate per-ticker data files for the Gemini baseline
    # (Other model variants would need their own NLP feature generation pipeline)
    prepare_ppo_data("gemini")