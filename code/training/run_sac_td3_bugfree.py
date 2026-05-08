#!/usr/bin/env python3
"""
Bug-free SAC & TD3 long-only training with pct_return reward.

BUG FIXES applied vs original train_sac.py:
  1. YahooDownloader: column-NAME mapping (not positional) + auto_adjust=True
     -> correct OHLC (before: open=AdjClose, high=Close, low=High, close=Open)
  2. NLP shift(1): prevents look-ahead bias (same as previous version)

EXPERIMENTS:
  Phase 1: SAC long-only pct_return x 10 seeds (fixed best params)
  Phase 2: TD3 long-only pct_return x 10 tuning iterations x 10 seeds each

Usage:
  # SAC 10-seed baseline
  python run_sac_td3_bugfree.py --algo sac --phase baseline

  # TD3 tuning iteration 0 (default params)
  python run_sac_td3_bugfree.py --algo td3 --phase tune --tune_iter 0

  # TD3 tuning iteration N
  python run_sac_td3_bugfree.py --algo td3 --phase tune --tune_iter N
"""
import pandas as pd
import numpy as np
import os
import sys
import json
import argparse
from datetime import datetime

# Ensure FinRL from the LOCAL PrimoGPT repo is imported (NOT pip finrl)
# INSERT FIRST so it overrides any pip-installed finrl
_PRIMOGPT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../PrimoGPT-main'))
if _PRIMOGPT_PATH not in sys.path:
    sys.path.insert(0, _PRIMOGPT_PATH)

from fixed_yahoodownloader import FixedYahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_primo_trading.env_primorl import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback

# ============================================================
# Fixed data pipeline settings
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

TICKERS = ["AAPL", "AMZN", "CRM", "MSFT", "NFLX"]
SEEDS = [42, 123, 456, 789, 2024, 55, 999, 314, 271, 1618]
INITIAL_AMOUNT = 100000

# ============================================================
# SAC best params (from previous experiments)
# ============================================================
SAC_BEST_PARAMS = {
    "batch_size": 256,
    "buffer_size": 100000,
    "learning_rate": 0.0001,
    "learning_starts": 2000,
    "ent_coef": "auto_0.1",
    "tau": 0.005,
    "gamma": 0.99,
}
SAC_POLICY_KWARGS = dict(net_arch=[64, 64], n_critics=2)
SAC_TOTAL_TIMESTEPS = 400000

# ============================================================
# TD3 tuning grid (10 iterations)
# ============================================================
TD3_TUNING_GRID = [
    # Iter 0: baseline (defaults)
    {"learning_rate": 0.0001, "tau": 0.005, "gamma": 0.99,
     "batch_size": 256, "buffer_size": 100000, "learning_starts": 2000,
     "policy_delay": 2, "action_noise": "normal",
     "net_arch": [64, 64]},
    # Iter 1: smaller net + slower LR
    {"learning_rate": 3e-5, "tau": 0.003, "gamma": 0.99,
     "batch_size": 256, "buffer_size": 100000, "learning_starts": 2000,
     "policy_delay": 2, "action_noise": "normal",
     "net_arch": [64, 64]},
    # Iter 2: larger net
    {"learning_rate": 0.0001, "tau": 0.005, "gamma": 0.99,
     "batch_size": 256, "buffer_size": 100000, "learning_starts": 2000,
     "policy_delay": 2, "action_noise": "normal",
     "net_arch": [128, 128]},
    # Iter 3: even larger net + more warmup
    {"learning_rate": 0.0001, "tau": 0.005, "gamma": 0.99,
     "batch_size": 256, "buffer_size": 100000, "learning_starts": 5000,
     "policy_delay": 2, "action_noise": "normal",
     "net_arch": [256, 256]},
    # Iter 4: slower policy update (delay=4) + slower LR
    {"learning_rate": 3e-5, "tau": 0.005, "gamma": 0.99,
     "batch_size": 256, "buffer_size": 100000, "learning_starts": 2000,
     "policy_delay": 4, "action_noise": "normal",
     "net_arch": [64, 64]},
    # Iter 5: higher LR + larger batch + OU noise
    {"learning_rate": 0.0003, "tau": 0.005, "gamma": 0.99,
     "batch_size": 512, "buffer_size": 200000, "learning_starts": 5000,
     "policy_delay": 2, "action_noise": "ornstein_uhlenbeck",
     "net_arch": [64, 64]},
    # Iter 6: small LR + small net + fast tau
    {"learning_rate": 1e-5, "tau": 0.01, "gamma": 0.99,
     "batch_size": 128, "buffer_size": 50000, "learning_starts": 1000,
     "policy_delay": 2, "action_noise": "normal",
     "net_arch": [32, 32]},
    # Iter 7: moderate LR + policy_delay=8 + deep net
    {"learning_rate": 5e-5, "tau": 0.005, "gamma": 0.995,
     "batch_size": 256, "buffer_size": 100000, "learning_starts": 3000,
     "policy_delay": 8, "action_noise": "normal",
     "net_arch": [64, 64, 32]},
    # Iter 8: SAC-mimicking (LR=1e-4, tau=0.003, net=[64,64], delay=2)
    {"learning_rate": 1e-4, "tau": 0.003, "gamma": 0.99,
     "batch_size": 256, "buffer_size": 100000, "learning_starts": 2000,
     "policy_delay": 2, "action_noise": "normal",
     "net_arch": [64, 64]},
    # Iter 9: aggressive exploration (high LR, OU noise, big buffer)
    {"learning_rate": 0.0003, "tau": 0.01, "gamma": 0.99,
     "batch_size": 512, "buffer_size": 200000, "learning_starts": 5000,
     "policy_delay": 4, "action_noise": "ornstein_uhlenbeck",
     "net_arch": [128, 128]},
]

TD3_TOTAL_TIMESTEPS = 400000


class SafeTrainingCallback(BaseCallback):
    """Clip gradients and detect NaN to prevent Q-function divergence."""
    def __init__(self, max_grad_norm=1.0, check_freq=2048, verbose=0):
        super().__init__(verbose)
        self.max_grad_norm = max_grad_norm
        self.check_freq = check_freq

    def _on_step(self):
        try:
            torch.nn.utils.clip_grad_norm_(self.model.actor.parameters(), self.max_grad_norm)
            for critic in [self.model.critic, self.model.critic_target]:
                torch.nn.utils.clip_grad_norm_(critic.parameters(), self.max_grad_norm)
        except Exception:
            pass
        if self.num_timesteps % self.check_freq == 0 and self.num_timesteps > 0:
            for name, param in self.model.policy.named_parameters():
                if param.isnan().any():
                    print(f"  WARNING: NaN in {name} at step {self.num_timesteps}")
                    return False
        return True


def load_data(ticker, data_path):
    """Load and merge OHLCV + NLP data with bug-free pipeline."""
    # 1. NLP features
    df_nlp = pd.read_csv(data_path)
    df_nlp = df_nlp.rename(columns={'Date': 'date'})
    if 'Adj Close Price' in df_nlp.columns:
        df_nlp = df_nlp.rename(columns={'Adj Close Price': 'close_nlp'})
    elif 'Close' in df_nlp.columns:
        df_nlp = df_nlp.rename(columns={'Close': 'close_nlp'})
    df_nlp = df_nlp.rename(columns=COLUMN_MAPPING)
    cols_to_drop = [c for c in ['Adj Close Price', 'Returns', 'Bin Label', 'Prompt', 'close_nlp'] if c in df_nlp.columns]
    df_nlp = df_nlp.drop(columns=cols_to_drop, errors='ignore')

    # 2. OHLCV via FIXED YahooDownloader (auto_adjust=True, column-name mapping)
    print(f"  Downloading OHLCV for {ticker} via FixedYahooDownloader...")
    df_yahoo = FixedYahooDownloader(
        start_date=DATA_DOWNLOAD_START,
        end_date=TRADE_END_DATE,
        ticker_list=[ticker]
    ).fetch_data()

    # 3. Technical indicators
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_vix=True,
        use_turbulence=False,
        user_defined_feature=False
    )
    processed = fe.preprocess_data(df_yahoo)

    # 4. Merge NLP
    processed['date'] = pd.to_datetime(processed['date']).dt.strftime('%Y-%m-%d')
    df_nlp['date'] = pd.to_datetime(df_nlp['date']).dt.strftime('%Y-%m-%d')
    processed_full = processed.merge(df_nlp, on='date', how='left')

    # 5. SHIFT(1) NLP features to prevent look-ahead bias
    for col in FUNDAMENTAL_INDICATORS:
        processed_full[col] = processed_full.groupby('tic')[col].shift(1)
    processed_full = processed_full.fillna(0)

    # 6. Handle duplicate columns from merge
    for col in ['close', 'tic']:
        if f'{col}_x' in processed_full.columns and f'{col}_y' in processed_full.columns:
            processed_full[col] = processed_full[f'{col}_x']
            processed_full = processed_full.drop(columns=[f'{col}_x', f'{col}_y'])

    # 7. Split
    train_data = data_split(processed_full, TRAIN_START_DATE, TRAIN_END_DATE)
    test_data = data_split(processed_full, TRADE_START_DATE, TRADE_END_DATE)
    print(f"  Train: {len(train_data)} rows | Test: {len(test_data)} rows")

    return train_data, test_data


def make_env(train_data, reward_type="pct_return"):
    """Create training environment."""
    stock_dimension = 1
    state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension + len(FUNDAMENTAL_INDICATORS)*stock_dimension
    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension

    env_kwargs = {
        "hmax": 1000,
        "initial_amount": INITIAL_AMOUNT,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "fundamental_indicator_list": FUNDAMENTAL_INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1,
        "reward_type": reward_type,
        "verbose": 0
    }

    e_train_gym = StockTradingEnv(df=train_data, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()
    return env_train, env_kwargs


def predict(model, test_data, env_kwargs):
    """Run out-of-sample evaluation with NaN clipping."""
    env_kwargs_copy = {**env_kwargs, "verbose": 0}
    e_trade_gym = StockTradingEnv(df=test_data, **env_kwargs_copy)
    test_env, test_obs = e_trade_gym.get_sb_env()
    test_env.reset()
    max_steps = len(e_trade_gym.df.index.unique()) - 1

    for i in range(len(e_trade_gym.df.index.unique())):
        action, _states = model.predict(test_obs, deterministic=True)
        if np.any(np.isnan(action)) or np.any(np.isinf(action)):
            print(f"  WARNING: NaN/Inf action at step {i}, clipping to 0")
            action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
        test_obs, rewards, dones, info = test_env.step(action)
        if i == max_steps - 1:
            account_memory = test_env.env_method(method_name="save_asset_memory")
            actions_memory = test_env.env_method(method_name="save_action_memory")
        if dones[0]:
            print("  hit end!")
            break

    df_account = account_memory[0]
    df_actions = actions_memory[0]
    final_value = df_account.iloc[-1, 1]
    ret_pct = (final_value - INITIAL_AMOUNT) / INITIAL_AMOUNT * 100
    return ret_pct, df_account, df_actions


def train_sac(ticker, data_path, seed, results_base):
    """Train SAC with best params on one ticker+seed."""
    print(f"\n{'='*60}")
    print(f"SAC | {ticker} | seed={seed}")
    print(f"{'='*60}")

    train_data, test_data = load_data(ticker, data_path)
    env_train, env_kwargs = make_env(train_data, reward_type="pct_return")

    agent = DRLAgent(env=env_train)
    model = agent.get_model("sac", model_kwargs=SAC_BEST_PARAMS,
                            policy_kwargs=SAC_POLICY_KWARGS, seed=seed)

    # Logging
    results_dir = f"{results_base}/seed{seed}/{ticker}"
    os.makedirs(results_dir, exist_ok=True)
    new_logger = configure(results_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    # Train
    safe_callback = SafeTrainingCallback(max_grad_norm=1.0, check_freq=2048)
    trained = model.learn(total_timesteps=SAC_TOTAL_TIMESTEPS,
                          tb_log_name='sac', callback=safe_callback)

    # Evaluate
    ret_pct, df_account, df_actions = predict(trained, test_data, env_kwargs)
    print(f"  -> {ticker} seed={seed}: {ret_pct:+.1f}%")

    # Save
    out_dir = f"{results_dir}/predictions"
    os.makedirs(out_dir, exist_ok=True)
    df_account.to_csv(f"{out_dir}/account_value.csv", index=False)
    df_actions.to_csv(f"{out_dir}/actions.csv", index=False)

    return ret_pct


def train_td3(ticker, data_path, seed, tune_params, results_base, tune_iter):
    """Train TD3 with given tuning params on one ticker+seed."""
    print(f"\n{'='*60}")
    print(f"TD3 tune_iter={tune_iter} | {ticker} | seed={seed}")
    print(f"{'='*60}")

    train_data, test_data = load_data(ticker, data_path)
    env_train, env_kwargs = make_env(train_data, reward_type="pct_return")

    # Extract params
    net_arch = tune_params.pop("net_arch", [64, 64])
    action_noise_type = tune_params.pop("action_noise", "normal")
    td3_params = {k: v for k, v in tune_params.items()}

    # Action noise
    from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
    n_actions = env_train.action_space.shape[0]
    if action_noise_type == "ou":
        action_noise = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    else:
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    td3_params["action_noise"] = action_noise
    policy_kwargs = dict(net_arch=net_arch, n_critics=2)

    agent = DRLAgent(env=env_train)
    model = agent.get_model("td3", model_kwargs=td3_params,
                            policy_kwargs=policy_kwargs, seed=seed)

    results_dir = f"{results_base}/tune{tune_iter}/seed{seed}/{ticker}"
    os.makedirs(results_dir, exist_ok=True)
    new_logger = configure(results_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    safe_callback = SafeTrainingCallback(max_grad_norm=1.0, check_freq=2048)
    trained = model.learn(total_timesteps=TD3_TOTAL_TIMESTEPS,
                          tb_log_name='td3', callback=safe_callback)

    ret_pct, df_account, df_actions = predict(trained, test_data, env_kwargs)
    print(f"  -> {ticker} seed={seed} tune{tune_iter}: {ret_pct:+.1f}%")

    out_dir = f"{results_dir}/predictions"
    os.makedirs(out_dir, exist_ok=True)
    df_account.to_csv(f"{out_dir}/account_value.csv", index=False)
    df_actions.to_csv(f"{out_dir}/actions.csv", index=False)

    return ret_pct


def run_sac_baseline(results_base):
    """Phase 1: SAC 10-seed baseline."""
    print("\n" + "="*70)
    print("PHASE 1: SAC long-only pct_return | 10 seeds")
    print("="*70)

    results = {}  # {ticker: {seed: ret_pct}}

    for seed in SEEDS:
        for ticker in TICKERS:
            data_path = os.path.abspath(os.path.join(
                os.path.dirname(__file__), f'../ppo_data/gemini_ppo_{ticker}_data.csv'))
            if not os.path.exists(data_path):
                print(f"  Data missing: {data_path}")
                continue
            try:
                ret = train_sac(ticker, data_path, seed, f"{results_base}/sac_baseline")
                if ticker not in results:
                    results[ticker] = {}
                results[ticker][seed] = ret
            except Exception as e:
                print(f"  ERROR: SAC failed for {ticker} seed={seed}: {e}")
                import traceback
                traceback.print_exc()

    # Summary
    print_summary("SAC baseline", results)
    save_summary(results_base, "sac_baseline", results, SAC_BEST_PARAMS)
    return results


def run_td3_tuning(results_base, tune_iter):
    """Phase 2: TD3 tuning - one iteration with 10 seeds."""
    tune_params = TD3_TUNING_GRID[tune_iter].copy()
    print("\n" + "="*70)
    print(f"PHASE 2: TD3 long-only pct_return | Tune iter {tune_iter}")
    print(f"  Params: {tune_params}")
    print("="*70)

    results = {}

    for seed in SEEDS:
        for ticker in TICKERS:
            data_path = os.path.abspath(os.path.join(
                os.path.dirname(__file__), f'../ppo_data/gemini_ppo_{ticker}_data.csv'))
            if not os.path.exists(data_path):
                continue
            try:
                ret = train_td3(ticker, data_path, seed, tune_params.copy(),
                                f"{results_base}/td3_tune", tune_iter)
                if ticker not in results:
                    results[ticker] = {}
                results[ticker][seed] = ret
            except Exception as e:
                print(f"  ERROR: TD3 failed for {ticker} seed={seed} tune={tune_iter}: {e}")
                import traceback
                traceback.print_exc()

    print_summary(f"TD3 tune_iter={tune_iter}", results)
    save_summary(results_base, f"td3_tune/iter{tune_iter}", results, TD3_TUNING_GRID[tune_iter])
    return results


def print_summary(label, results):
    """Print results table."""
    print(f"\n{'='*70}")
    print(f"RESULTS: {label}")
    print(f"{'='*70}")
    header = f"{'Seed':<10}" + "".join(f"{t:>8}" for t in TICKERS)
    print(header)
    print("-" * len(header))

    all_seeds = set()
    for t in TICKERS:
        if t in results:
            all_seeds.update(results[t].keys())
    all_seeds = sorted(all_seeds)

    for s in all_seeds:
        row = f"seed{s:<6}" + "".join(
            f"{results[t].get(s, 0):>+8.1f}" if t in results and s in results[t] else f"{'---':>8}"
            for t in TICKERS)
        print(row)

    # Means
    means = {}
    for t in TICKERS:
        if t in results and results[t]:
            means[t] = np.mean(list(results[t].values()))
        else:
            means[t] = None
    stds = {}
    for t in TICKERS:
        if t in results and len(results[t]) > 1:
            stds[t] = np.std(list(results[t].values()), ddof=1)
        else:
            stds[t] = 0

    row = f"{'MEAN':<10}" + "".join(f"{means[t]:>+8.1f}" if means[t] is not None else f"{'---':>8}" for t in TICKERS)
    print(row)
    row = f"{'STD':<10}" + "".join(f"{stds[t]:>8.1f}" if stds[t] else f"{'---':>8}" for t in TICKERS)
    print(row)
    overall = np.mean([m for m in means.values() if m is not None])
    print(f"\n  Overall mean: {overall:+.1f}%")


def save_summary(results_base, label, results, params):
    """Save results summary to JSON."""
    summary = {
        "label": label,
        "timestamp": datetime.now().isoformat(),
        "params": {k: str(v) for k, v in params.items()},
        "results": {t: {str(s): v for s, v in seeds.items()} for t, seeds in results.items()},
        "means": {t: float(np.mean(list(v.values()))) for t, v in results.items() if v},
        "stds": {t: float(np.std(list(v.values()), ddof=1)) for t, v in results.items() if len(v) > 1},
    }
    path = f"{results_base}/{label}_summary.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary: {path}")


if __name__ == '__main__':
    import torch  # needed by SafeTrainingCallback
    parser = argparse.ArgumentParser(description='Bug-free SAC/TD3 long-only training')
    parser.add_argument('--algo', type=str, default='sac', choices=['sac', 'td3'])
    parser.add_argument('--phase', type=str, default='baseline',
                        choices=['baseline', 'tune'],
                        help='baseline=SAC 10-seed, tune=TD3 tuning iteration')
    parser.add_argument('--tune_iter', type=int, default=0,
                        help='TD3 tuning iteration (0-9)')
    parser.add_argument('--results_dir', type=str, default='bugfree_results',
                        help='Base directory for results')
    args = parser.parse_args()

    results_base = os.path.abspath(os.path.join(os.path.dirname(__file__), args.results_dir))
    os.makedirs(results_base, exist_ok=True)

    if args.algo == 'sac' and args.phase == 'baseline':
        run_sac_baseline(results_base)
    elif args.algo == 'td3' and args.phase == 'tune':
        if args.tune_iter >= len(TD3_TUNING_GRID):
            print(f"ERROR: tune_iter={args.tune_iter} out of range [0, {len(TD3_TUNING_GRID)-1}]")
            sys.exit(1)
        run_td3_tuning(results_base, args.tune_iter)
    else:
        print(f"Invalid combo: algo={args.algo} phase={args.phase}")
        sys.exit(1)
