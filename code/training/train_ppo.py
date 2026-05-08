import pandas as pd
import numpy as np
import os
import sys

# Ensure FinRL from the original PrimoGPT repo can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../PrimoGPT-main')))

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_primo_trading.env_primorl import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.logger import configure

# ============================================================
# KNOWN LIMITATIONS & BIAS DISCLOSURES
# ============================================================
# 1. EXECUTION TIMING: The environment assumes the agent observes day T's close
#    price and executes trades at that same close price. In practice, the close
#    price is only known after market hours, making same-close execution impossible.
#    This optimistic assumption inflates backtested returns. A more realistic model
#    would execute at T+1 open or add a slippage model (5-10 bps).
#    -> NOT FIXED: This is a structural limitation of FinRL's StockTradingEnv
#       and the reference code shares the same assumption. Document in paper.
#
# 2. SURVIVORSHIP BIAS: The tickers (AAPL, AMZN, CRM, MSFT, NFLX) are all
#    currently active mega-cap tech stocks. No delisted or failed companies are
#    included, so the model only learns from successful stocks, overestimating
#    potential returns. Document as a known limitation.
#
# 3. TRANSACTION COSTS: We use 0.1% (buy_cost_pct=0.001), which is at the low
#    end of realistic but does not account for slippage, bid-ask spread, or
#    market impact. The reference code uses 0%. Our 0.1% is a deliberate choice.
# ============================================================

# ============================================================
# Reference-matched settings (from PrimoGPT notebooks)
# ============================================================
DATA_DOWNLOAD_START = '2022-04-01'  # Download from earlier date to allow warmup period
TRAIN_START_DATE = '2022-04-01'     # Matching paper: was '2022-07-01' (we skipped warmup), now aligned
TRAIN_END_DATE = '2024-07-31'        # Fixed: was '2024-08-01', reference uses '2024-07-31'
TRADE_START_DATE = '2024-08-01'
TRADE_END_DATE = '2025-02-28'

INDICATORS = [
    'macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30',
    'dx_30', 'close_30_sma', 'close_60_sma'
]

# snake_case names matching the reference config.py FUNDAMENTAL_INDICATORS
FUNDAMENTAL_INDICATORS = [
    'news_relevance', 'sentiment', 'price_impact_potential',
    'trend_direction', 'earnings_impact',
    'investor_confidence', 'risk_profile_change'
]

# Title Case to snake_case column mapping (matching reference notebook Cell 15)
COLUMN_MAPPING = {
    'News Relevance': 'news_relevance',
    'Sentiment': 'sentiment',
    'Price Impact Potential': 'price_impact_potential',
    'Trend Direction': 'trend_direction',
    'Earnings Impact': 'earnings_impact',
    'Investor Confidence': 'investor_confidence',
    'Risk Profile Change': 'risk_profile_change'
}


def train_and_eval_ppo(model_id, ticker, data_path, reward_type="dollar_delta", seed=None):
    print(f"--- Training & Eval PPO for {ticker} using {model_id} NLP Features | Reward: {reward_type} | Seed: {seed} ---")
    
    # 1. LOAD NLP FEATURES FROM MODEL-SPECIFIC CSV
    df_nlp = pd.read_csv(data_path)
    
    # Rename Title Case NLP columns to snake_case (matching reference notebook Cell 15)
    df_nlp = df_nlp.rename(columns={'Date': 'date'})
    if 'Adj Close Price' in df_nlp.columns:
        df_nlp = df_nlp.rename(columns={'Adj Close Price': 'close'})
    elif 'Close' in df_nlp.columns:
        df_nlp = df_nlp.rename(columns={'Close': 'close'})
    
    # Apply the Title Case -> snake_case mapping for NLP features
    df_nlp = df_nlp.rename(columns=COLUMN_MAPPING)
    
    # Drop columns we don't need in the final DataFrame
    columns_to_drop = [c for c in ['Adj Close Price', 'Returns', 'Bin Label', 'Prompt'] if c in df_nlp.columns]
    df_nlp = df_nlp.drop(columns=columns_to_drop, errors='ignore')
    
    # 2. DOWNLOAD FULL OHLCV DATA VIA YahooDownloader (matches reference exactly)
    # NOTE: YahooDownloader has a known column rename bug with yfinance >= 0.2.x where
    # columns come back alphabetically [Adj Close, Close, High, Low, Open, Volume] instead
    # of the expected [Open, High, Low, Close, Adj Close, Volume]. This causes "close" to
    # be mapped to the raw Open price instead of Adj Close. The reference paper was produced
    # with this bug, so we use YahooDownloader to match their results exactly.
    print(f"Downloading OHLCV data for {ticker} via YahooDownloader...")
    df_yahoo = YahooDownloader(
        start_date=DATA_DOWNLOAD_START,
        end_date=TRADE_END_DATE,
        ticker_list=[ticker]
    ).fetch_data()
    
    print(f"  Downloaded {len(df_yahoo)} rows of OHLCV data for {ticker}")
    print(f"  Columns: {df_yahoo.columns.tolist()}")
    
    # 3. ADD TECHNICAL INDICATORS VIA FINRL FeatureEngineer
    # Now we have real OHLCV data, so VIX can be enabled (matching reference)
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_vix=True,                  # ENABLED: matching reference notebook
        use_turbulence=False,          # Single stock, no turbulence
        user_defined_feature=False
    )
    
    processed = fe.preprocess_data(df_yahoo)
    print(f"  After FeatureEngineer: {len(processed)} rows, columns: {processed.columns.tolist()}")
    
    # 4. MERGE NLP FEATURES WITH OHLCV+TECHNICAL DATA (matching reference notebook Cell 15)
    # Ensure date columns are in the same format for merging
    processed['date'] = pd.to_datetime(processed['date']).dt.strftime('%Y-%m-%d')
    df_nlp['date'] = pd.to_datetime(df_nlp['date']).dt.strftime('%Y-%m-%d')
    
    # Merge on date
    processed_full = processed.merge(df_nlp, on='date', how='left')
    
    # ---- NLP FEATURE SHIFT: ENABLED (bias fix) ----
    # Shifts NLP features by 1 day so that after-market/overnight news from date T
    # is used for date T+1's trading decision. This eliminates look-ahead bias:
    # without the shift, the agent at date T sees news published AFTER 4PM on date T,
    # which was not available when the close price was set.
    for col in FUNDAMENTAL_INDICATORS:
        processed_full[col] = processed_full.groupby('tic')[col].shift(1)
    
    processed_full = processed_full.fillna(0)
    
    # Remove any duplicate columns from the merge (e.g., close_x, close_y)
    # Keep the YahooDownloader close price as the authoritative source
    # (matches reference: reference drops 'Adj Close Price' from NLP CSV, keeping
    # YahooDownloader's 'close' which, due to the bug, is the raw Open price)
    for col in ['close', 'tic']:
        if f'{col}_x' in processed_full.columns and f'{col}_y' in processed_full.columns:
            processed_full[col] = processed_full[f'{col}_x']
            processed_full = processed_full.drop(columns=[f'{col}_x', f'{col}_y'])
    
    print(f"  After merge: {len(processed_full)} rows, {len(processed_full.columns)} columns")
    print(f"  Columns: {processed_full.columns.tolist()}")
    
    # ---- WARMUP ROW HANDLING ----
    # The first ~60 rows have unreliable technical indicator values (e.g., close_60_sma
    # needs 60 days, rsi_30 starts at 100, macd starts at 0). The paper uses bfill()
    # (which leaks future data) and keeps all rows. We use fillna(0) and keep all rows
    # to match the paper's training data size. The early 0-value indicators are a
    # trade-off: no future data leakage but noisier early observations.
    print(f"  Total rows (no warmup drop, matching paper): {len(processed_full)}")
    
    # 5. CHRONOLOGICAL DATA SPLIT (matching reference dates exactly)
    train_data = data_split(processed_full, TRAIN_START_DATE, TRAIN_END_DATE)
    test_data = data_split(processed_full, TRADE_START_DATE, TRADE_END_DATE)
    
    print(f"  Train: {len(train_data)} rows ({TRAIN_START_DATE} to {TRAIN_END_DATE})")
    print(f"  Test:  {len(test_data)} rows ({TRADE_START_DATE} to {TRADE_END_DATE})")
    
    # 6. ENVIRONMENT STATE SPACE DECLARATION
    # State: [cash] + [close, holdings] * stock_dim + tech_indicators * stock_dim + NLP_features * stock_dim
    # Note: VIX is downloaded and added to the DataFrame (use_vix=True) for data completeness,
    # but it is NOT included in the observation vector since it's not in tech_indicator_list.
    # This matches the reference notebook's state_space formula.
    stock_dimension = 1  # Single stock trading at a time
    # = 1 (cash) + 2*1 (price+holdings) + 8*1 (tech) + 7*1 (NLP) = 18
    state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension + len(FUNDAMENTAL_INDICATORS)*stock_dimension
    
    # Transaction costs kept at 0.1% (our deliberate choice per user)
    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension
    
    # Environment parameters matching reference notebook
    env_kwargs = {
        "hmax": 1000,                  # Maximum shares per action (matching reference)
        "initial_amount": 100000,      # Starting capital: $100,000 (matching reference)
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "fundamental_indicator_list": FUNDAMENTAL_INDICATORS,  # snake_case, matching reference
        "action_space": stock_dimension,
        "reward_scaling": 100,
        "reward_type": reward_type,    # Reward function variant
        "verbose": 0
    }
    
    # ---- 7. RL AGENT TRAINING (PPO) ----
    e_train_gym = StockTradingEnv(df=train_data, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()
    
    agent = DRLAgent(env=env_train)
    # Reference-matched PPO hyperparameters (from config.py PPO_PARAMS)
    PPO_PARAMS = {
        "n_steps": 2048,
        "ent_coef": 0.01,
        "learning_rate": 0.00025,
        "batch_size": 128,            # Matching reference notebook & config.py
    }
    # Pass seed through get_model's own parameter (not model_kwargs) to avoid
    # "multiple values for keyword argument" conflict since get_model also passes seed
    model_ppo = agent.get_model("ppo", model_kwargs=PPO_PARAMS, seed=seed)
    
    # Establish logging directories (include reward_type and seed in path)
    seed_str = f"_seed{seed}" if seed is not None else ""
    results_dir = f"ppo_results/{model_id}_{reward_type}{seed_str}/{ticker}"
    os.makedirs(results_dir, exist_ok=True)
    new_logger_ppo = configure(results_dir, ["stdout", "csv", "tensorboard"])
    model_ppo.set_logger(new_logger_ppo)
    
    # Execute PPO training with reference-matched timesteps (400K, matching reference notebook)
    print(f"Starting PPO training for {model_id} on {ticker}...")
    trained_ppo = agent.train_model(model=model_ppo, tb_log_name='ppo', total_timesteps=400000)
    
    # ---- 8. OUT-OF-SAMPLE EVALUATION (Testing) ----
    print(f"Starting Prediction/Evaluation for {model_id} on {ticker}...")
    env_kwargs["verbose"] = 0
    
    e_trade_gym = StockTradingEnv(df=test_data, **env_kwargs)
    
    df_account_value, df_actions = DRLAgent.DRL_prediction(
        model=trained_ppo, 
        environment=e_trade_gym
    )
    
    # ---- 9. SAVE RESULTS ----
    output_dir = f"ppo_results/{model_id}_{reward_type}{seed_str}/{ticker}/predictions"
    os.makedirs(output_dir, exist_ok=True)
    df_account_value.to_csv(f"{output_dir}/account_value.csv", index=False)
    df_actions.to_csv(f"{output_dir}/actions.csv", index=False)
    print(f"Saved evaluation results to {output_dir}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train PPO with different reward functions and seeds')
    parser.add_argument('--reward_type', type=str, default='pct_return',
                        choices=['dollar_delta', 'cash_penalty', 'pct_return', 
                                 'diff_sharpe', 'sortino', 'mean_variance', 
                                 'combined_paper', 'pct_risk_penalty'],
                        help='Reward function type to use')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    args = parser.parse_args()
    
    tickers = ["AAPL", "AMZN", "CRM", "MSFT", "NFLX"]
    
    for ticker in tickers:
        data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f'../ppo_data/gemini_ppo_{ticker}_data.csv'))
        if os.path.exists(data_path):
            train_and_eval_ppo("gemini", ticker, data_path, reward_type=args.reward_type, seed=args.seed)
        else:
            print(f"Data missing at: {data_path}")