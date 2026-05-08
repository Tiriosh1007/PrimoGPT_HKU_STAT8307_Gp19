"""
PPO Training with NLP Features — Gemini & QALoRA
=================================================
Key fixes vs previous SAC runs:
1. reward_scaling=100 (reference paper value; was 1 in our SAC runs)
2. VecNormalize wrapper for observation normalization (fixes scale problem)
3. PPO algorithm (reference paper's primary model)
4. Reference-matching hyperparameters

Supports: --mode long|short --llm gemini|qalora --seed N
"""
import pandas as pd
import numpy as np
import os, sys, argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../PrimoGPT-main')))

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# ============================================================
# CONFIG
# ============================================================
DATA_DOWNLOAD_START = '2022-04-01'
TRAIN_START_DATE = '2022-04-01'
TRAIN_END_DATE = '2024-07-31'
TRADE_START_DATE = '2024-08-01'
TRADE_END_DATE = '2025-02-28'

INDICATORS = [
    'macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30',
    'dx_30', 'close_30_sma', 'close_60_sma'
]

FUNDAMENTAL_INDICATORS = [
    'news_relevance', 'sentiment', 'price_impact_potential',
    'trend_direction', 'earnings_impact',
    'investor_confidence', 'risk_profile_change'
]

COLUMN_MAPPING = {
    'News Relevance': 'news_relevance',
    'Sentiment': 'sentiment',
    'Price Impact Potential': 'price_impact_potential',
    'Trend Direction': 'trend_direction',
    'Earnings Impact': 'earnings_impact',
    'Investor Confidence': 'investor_confidence',
    'Risk Profile Change': 'risk_profile_change'
}


def load_and_prepare_data(ticker, llm='gemini'):
    """Load price + NLP data, merge, shift NLP features."""
    # NLP data paths
    qalora_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../LLM_data_qalora'))
    ppo_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../ppo_data'))

    if llm == 'qalora':
        if ticker == 'AAPL':
            nlp_path = os.path.join(ppo_data_dir, 'qwen_qalora_ppo_data.csv')
        else:
            nlp_path = os.path.join(qalora_dir, f'qwen_qalora_ppo_{ticker}_data.csv')
    else:  # gemini
        nlp_path = os.path.join(ppo_data_dir, f'gemini_ppo_{ticker}_data.csv')

    # Load NLP
    df_nlp = pd.read_csv(nlp_path)
    df_nlp = df_nlp.rename(columns={'Date': 'date'})
    if 'Adj Close Price' in df_nlp.columns:
        df_nlp = df_nlp.rename(columns={'Adj Close Price': 'nlp_close'})
    df_nlp = df_nlp.rename(columns=COLUMN_MAPPING)
    drop_cols = [c for c in ['Returns', 'Bin Label', 'Prompt', 'ticker'] if c in df_nlp.columns]
    df_nlp = df_nlp.drop(columns=drop_cols, errors='ignore')

    # Load price data
    df_yahoo = YahooDownloader(
        start_date=DATA_DOWNLOAD_START,
        end_date=TRADE_END_DATE,
        ticker_list=[ticker]
    ).fetch_data()

    # Add technical indicators
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_vix=True,
        use_turbulence=False,
        user_defined_feature=False
    )
    processed = fe.preprocess_data(df_yahoo)

    # Merge
    processed['date'] = pd.to_datetime(processed['date']).dt.strftime('%Y-%m-%d')
    df_nlp['date'] = pd.to_datetime(df_nlp['date']).dt.strftime('%Y-%m-%d')

    # Use suffixes to avoid close column collision
    processed_full = processed.merge(df_nlp, on='date', how='left', suffixes=('', '_nlp'))

    # Drop the NLP close (we keep the YahooDownloader close)
    if 'nlp_close' in processed_full.columns:
        processed_full = processed_full.drop(columns=['nlp_close'])
    # Also clean up any other _nlp suffix duplicates
    for col in list(processed_full.columns):
        if col.endswith('_nlp'):
            processed_full = processed_full.drop(columns=[col])

    # NLP shift(1) for no look-ahead bias
    for col in FUNDAMENTAL_INDICATORS:
        processed_full[col] = processed_full.groupby('tic')[col].shift(1)
    processed_full = processed_full.fillna(0)

    # Split
    train_data = data_split(processed_full, TRAIN_START_DATE, TRAIN_END_DATE)
    test_data = data_split(processed_full, TRADE_START_DATE, TRADE_END_DATE)
    print(f"  Train: {len(train_data)} rows | Test: {len(test_data)} rows")

    return train_data, test_data


def train_and_eval_ppo(llm, ticker, mode='long', seed=None):
    """Train PPO with observation normalization and reference-matching params."""

    print(f"\n--- PPO {mode} for {ticker} | LLM: {llm} | Seed: {seed} ---")

    # Load data
    train_data, test_data = load_and_prepare_data(ticker, llm=llm)

    # Environment setup
    stock_dimension = 1

    if mode == 'long':
        from finrl.meta.env_primo_trading.env_primorl import StockTradingEnv
        state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension + len(FUNDAMENTAL_INDICATORS)*stock_dimension
        env_kwargs = {
            "hmax": 1000,
            "initial_amount": 100000,
            "num_stock_shares": [0] * stock_dimension,
            "buy_cost_pct": [0.001] * stock_dimension,
            "sell_cost_pct": [0.001] * stock_dimension,
            "state_space": state_space,
            "stock_dim": stock_dimension,
            "tech_indicator_list": INDICATORS,
            "fundamental_indicator_list": FUNDAMENTAL_INDICATORS,
            "action_space": stock_dimension,
            "reward_scaling": 100,  # Reference paper value! (was 1 in our SAC)
            "reward_type": "dollar_delta",  # Reference uses dollar_delta with scaling=100
            "verbose": 0,
        }
    elif mode == 'short':
        from finrl.meta.env_primo_trading.env_primorl_short_v2 import StockTradingEnv
        state_space = 1 + 2*stock_dimension + stock_dimension + len(INDICATORS)*stock_dimension + len(FUNDAMENTAL_INDICATORS)*stock_dimension
        env_kwargs = {
            "hmax": 1000,
            "initial_amount": 100000,
            "num_stock_shares": [0] * stock_dimension,
            "buy_cost_pct": [0.001] * stock_dimension,
            "sell_cost_pct": [0.001] * stock_dimension,
            "state_space": state_space,
            "stock_dim": stock_dimension,
            "tech_indicator_list": INDICATORS,
            "fundamental_indicator_list": FUNDAMENTAL_INDICATORS,
            "action_space": stock_dimension,
            "reward_scaling": 100,
            "reward_type": "dollar_delta",
            "verbose": 0,
            "short_selling": True,
            "hmax_short": 1000,
            "short_borrow_rate": 0.01,
            "margin_requirement": 0.5,
            "short_stop_loss_pct": 0.15,
            "max_short_days": 20,
            "borrow_escalation_rate": 0.5,
            "borrow_escalation_period": 10,
            "drawdown_penalty_weight": 2.0,
            "short_loss_penalty_weight": 1.5,
            "max_position_pct": 0.5,
        }

    # Create training env with VecNormalize for observation normalization
    e_train_gym = StockTradingEnv(df=train_data, **env_kwargs)
    env_train = DummyVecEnv([lambda: e_train_gym])
    # KEY FIX: Normalize observations so NLP features are visible to the network
    # clip_obs prevents outliers, norm_obs=True enables running mean/std normalization
    env_train = VecNormalize(env_train, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # PPO Agent — reference paper hyperparameters
    agent = DRLAgent(env=env_train)
    PPO_PARAMS = {
        "n_steps": 2048,
        "ent_coef": 0.01,
        "learning_rate": 0.00025,
        "batch_size": 128,
    }
    policy_kwargs = dict(net_arch=[64, 64])

    model_ppo = agent.get_model("ppo", model_kwargs=PPO_PARAMS,
                                 policy_kwargs=policy_kwargs, seed=seed)

    # Logging
    seed_str = f"_seed{seed}" if seed is not None else ""
    results_dir = f"ppo_results/{llm}_{mode}{seed_str}/{ticker}"
    os.makedirs(results_dir, exist_ok=True)
    new_logger = configure(results_dir, ["stdout", "csv", "tensorboard"])
    model_ppo.set_logger(new_logger)

    # Train
    print(f"Starting PPO {mode} training for {llm} on {ticker}...")
    trained_ppo = model_ppo.learn(
        total_timesteps=400000,
        tb_log_name='ppo',
    )

    # Evaluation — need VecNormalize for test env too
    print(f"Starting Prediction for {llm} on {ticker}...")
    env_kwargs["verbose"] = 0
    e_trade_gym = StockTradingEnv(df=test_data, **env_kwargs)
    test_env = DummyVecEnv([lambda: e_trade_gym])
    # DON'T retrain normalization on test data — use training stats
    test_env = VecNormalize(test_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)
    # Copy normalization stats from training env
    test_env.obs_rms = env_train.obs_rms
    test_env.clip_obs = env_train.clip_obs

    test_obs = test_env.reset()
    max_steps = len(test_data.index.unique()) - 1

    for i in range(len(test_data.index.unique())):
        action, _states = trained_ppo.predict(test_obs, deterministic=True)
        if np.any(np.isnan(action)) or np.any(np.isinf(action)):
            print(f"  WARNING: NaN/Inf action at step {i}")
            action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
        test_obs, rewards, dones, info = test_env.step(action)
        if i == max_steps - 1:
            # Call env_method on VecNormalize wrapper (which delegates to inner envs)
            account_memory = test_env.env_method(method_name="save_asset_memory")
            actions_memory = test_env.env_method(method_name="save_action_memory")
        if dones[0]:
            print("hit end!")
            break

    df_account_value = account_memory[0]
    df_actions = actions_memory[0]

    # Save results
    output_dir = f"ppo_results/{llm}_{mode}{seed_str}/{ticker}/predictions"
    os.makedirs(output_dir, exist_ok=True)
    df_account_value.to_csv(f"{output_dir}/account_value.csv", index=False)
    df_actions.to_csv(f"{output_dir}/actions.csv", index=False)

    final_val = df_account_value.iloc[-1, 1]
    ret = (final_val - 100000) / 100000 * 100
    print(f"Saved evaluation to {output_dir}")
    print(f"  Final value: {final_val:.2f}, Return: {ret:+.2f}%")

    # Save VecNormalize stats for reproducibility
    env_train.save(f"{output_dir}/vecnormalize_stats.pkl")
    print(f"  VecNormalize stats saved")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train PPO with NLP features')
    parser.add_argument('--mode', type=str, default='long', choices=['long', 'short'])
    parser.add_argument('--llm', type=str, default='gemini', choices=['gemini', 'qalora'])
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    tickers = ["AAPL", "AMZN", "CRM", "MSFT", "NFLX"]

    for ticker in tickers:
        try:
            train_and_eval_ppo(args.llm, ticker, mode=args.mode, seed=args.seed)
        except Exception as e:
            print(f"ERROR: PPO {args.mode} failed for {ticker} with seed {args.seed}: {e}")
            import traceback
            traceback.print_exc()
