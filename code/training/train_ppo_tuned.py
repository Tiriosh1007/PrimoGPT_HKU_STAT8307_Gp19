"""
PPO Training — Hyperparameter Tuned Version (QALoRA + Long Only)
================================================================
Tuning changes from baseline:
1. Larger network: [128, 128] instead of [64, 64] — more capacity to learn NLP patterns
2. Lower learning rate: 1e-4 instead of 2.5e-4 — more stable convergence
3. Smaller n_steps: 1024 instead of 2048 — more frequent policy updates
4. Higher entropy: 0.02 instead of 0.01 — more exploration of NLP-driven strategies
5. Larger batch: 256 instead of 128 — more stable gradient estimates
6. Total timesteps: 500K instead of 400K — more training time
7. reward_scaling=100, VecNormalize (same as before)

Usage: python3 train_ppo_tuned.py --seed N
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

# ============================================================
# TUNED HYPERPARAMETERS
# ============================================================
PPO_PARAMS_TUNED = {
    "n_steps": 1024,           # was 2048 — more frequent updates
    "ent_coef": 0.02,          # was 0.01 — more exploration
    "learning_rate": 0.0001,   # was 0.00025 — more stable
    "batch_size": 256,         # was 128 — better gradient estimates
}
POLICY_KWARGS_TUNED = dict(net_arch=[128, 128])  # was [64, 64]
TOTAL_TIMESTEPS = 500000      # was 400000


def load_and_prepare_data(ticker, llm='qalora'):
    """Load price + NLP data, merge, shift NLP features."""
    qalora_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../LLM_data_qalora'))
    ppo_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../ppo_data'))

    if llm == 'qalora':
        if ticker == 'AAPL':
            nlp_path = os.path.join(ppo_data_dir, 'qwen_qalora_ppo_data.csv')
        else:
            nlp_path = os.path.join(qalora_dir, f'qwen_qalora_ppo_{ticker}_data.csv')
    else:
        nlp_path = os.path.join(ppo_data_dir, f'gemini_ppo_{ticker}_data.csv')

    df_nlp = pd.read_csv(nlp_path)
    df_nlp = df_nlp.rename(columns={'Date': 'date'})
    if 'Adj Close Price' in df_nlp.columns:
        df_nlp = df_nlp.rename(columns={'Adj Close Price': 'nlp_close'})
    df_nlp = df_nlp.rename(columns=COLUMN_MAPPING)
    drop_cols = [c for c in ['Returns', 'Bin Label', 'Prompt', 'ticker'] if c in df_nlp.columns]
    df_nlp = df_nlp.drop(columns=drop_cols, errors='ignore')

    df_yahoo = YahooDownloader(
        start_date=DATA_DOWNLOAD_START,
        end_date=TRADE_END_DATE,
        ticker_list=[ticker]
    ).fetch_data()

    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_vix=True,
        use_turbulence=False,
        user_defined_feature=False
    )
    processed = fe.preprocess_data(df_yahoo)

    processed['date'] = pd.to_datetime(processed['date']).dt.strftime('%Y-%m-%d')
    df_nlp['date'] = pd.to_datetime(df_nlp['date']).dt.strftime('%Y-%m-%d')

    processed_full = processed.merge(df_nlp, on='date', how='left', suffixes=('', '_nlp'))
    if 'nlp_close' in processed_full.columns:
        processed_full = processed_full.drop(columns=['nlp_close'])
    for col in list(processed_full.columns):
        if col.endswith('_nlp'):
            processed_full = processed_full.drop(columns=[col])

    # NLP shift(1) for no look-ahead bias
    for col in FUNDAMENTAL_INDICATORS:
        processed_full[col] = processed_full.groupby('tic')[col].shift(1)
    processed_full = processed_full.fillna(0)

    train_data = data_split(processed_full, TRAIN_START_DATE, TRAIN_END_DATE)
    test_data = data_split(processed_full, TRADE_START_DATE, TRADE_END_DATE)
    print(f"  Train: {len(train_data)} rows | Test: {len(test_data)} rows")

    return train_data, test_data


def train_and_eval_ppo(ticker, seed=None):
    """Train PPO with TUNED hyperparameters + QALoRA + Long Only."""

    print(f"\n--- PPO TUNED Long for {ticker} | QALoRA | Seed: {seed} ---")

    train_data, test_data = load_and_prepare_data(ticker, llm='qalora')

    from finrl.meta.env_primo_trading.env_primorl import StockTradingEnv

    stock_dimension = 1
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
        "reward_scaling": 100,
        "reward_type": "dollar_delta",
        "verbose": 0,
    }

    # Training env with VecNormalize
    e_train_gym = StockTradingEnv(df=train_data, **env_kwargs)
    env_train = DummyVecEnv([lambda: e_train_gym])
    env_train = VecNormalize(env_train, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # PPO with TUNED hyperparameters
    agent = DRLAgent(env=env_train)
    model_ppo = agent.get_model("ppo", model_kwargs=PPO_PARAMS_TUNED,
                                 policy_kwargs=POLICY_KWARGS_TUNED, seed=seed)

    seed_str = f"_seed{seed}" if seed is not None else ""
    results_dir = f"ppo_results/qalora_long_tuned{seed_str}/{ticker}"
    os.makedirs(results_dir, exist_ok=True)
    new_logger = configure(results_dir, ["stdout", "csv", "tensorboard"])
    model_ppo.set_logger(new_logger)

    print(f"Starting PPO TUNED training for QALoRA on {ticker}...")
    print(f"  Params: n_steps={PPO_PARAMS_TUNED['n_steps']}, ent={PPO_PARAMS_TUNED['ent_coef']}, "
          f"lr={PPO_PARAMS_TUNED['learning_rate']}, batch={PPO_PARAMS_TUNED['batch_size']}")
    print(f"  Network: {POLICY_KWARGS_TUNED['net_arch']}, Steps: {TOTAL_TIMESTEPS}")

    trained_ppo = model_ppo.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        tb_log_name='ppo_tuned',
    )

    # Evaluation
    print(f"Starting Prediction for QALoRA on {ticker}...")
    env_kwargs["verbose"] = 0
    e_trade_gym = StockTradingEnv(df=test_data, **env_kwargs)
    test_env = DummyVecEnv([lambda: e_trade_gym])
    test_env = VecNormalize(test_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)
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
            account_memory = test_env.env_method(method_name="save_asset_memory")
            actions_memory = test_env.env_method(method_name="save_action_memory")
        if dones[0]:
            print("hit end!")
            break

    df_account_value = account_memory[0]
    df_actions = actions_memory[0]

    output_dir = f"ppo_results/qalora_long_tuned{seed_str}/{ticker}/predictions"
    os.makedirs(output_dir, exist_ok=True)
    df_account_value.to_csv(f"{output_dir}/account_value.csv", index=False)
    df_actions.to_csv(f"{output_dir}/actions.csv", index=False)

    final_val = df_account_value.iloc[-1, 1]
    ret = (final_val - 100000) / 100000 * 100
    print(f"Saved evaluation to {output_dir}")
    print(f"  Final value: {final_val:.2f}, Return: {ret:+.2f}%")

    env_train.save(f"{output_dir}/vecnormalize_stats.pkl")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train PPO TUNED with QALoRA features (Long Only)')
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    tickers = ["AAPL", "AMZN", "CRM", "MSFT", "NFLX"]

    for ticker in tickers:
        try:
            train_and_eval_ppo(ticker, seed=args.seed)
        except Exception as e:
            print(f"ERROR: PPO TUNED failed for {ticker} with seed {args.seed}: {e}")
            import traceback
            traceback.print_exc()
