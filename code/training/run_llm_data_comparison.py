#!/usr/bin/env python3
"""
LLM Data Impact Test — PPO with Normalized Observations
========================================================
Purpose: Compare trading performance under different LLM data sources.

The previous experiment failed because raw observations have severe scale mismatch:
  - cash ~100,000, price ~200, NLP features ~[-2,2]
  - NLP features contributed only 0.8% to policy decisions

This version adds z-score normalization to the observation space so that
NLP features are on comparable scale with other features.

5 LLM data sources:
  1. Gemini 3.1 Pro
  2. Qwen3.5-27B base
  3. Qwen3.5-27B + QLoRA
  4. Qwen3.5-27B + QA-LoRA
  5. No-NLP (tech indicators only — ablation baseline)

Controls:
  - Same PPO hyperparameters for all configs
  - Same price/tech data for all configs
  - Same seeds for all configs
  - Only difference = which LLM data feeds the NLP feature columns
"""

import pandas as pd
import numpy as np
import gymnasium as gym
import os, sys, json, time, warnings
from datetime import datetime
from pathlib import Path
from copy import deepcopy

warnings.filterwarnings("ignore")

# ── Paths ──
PPO_TRAIN = Path(__file__).resolve().parent
OUR_CODE = PPO_TRAIN.parent
PROJECT = OUR_CODE.parent
PRIMO_GPT = PROJECT / "PrimoGPT-main"

sys.path.insert(0, str(PRIMO_GPT))
sys.path.insert(0, str(PPO_TRAIN))
sys.path.insert(0, str(OUR_CODE))

from fixed_yahoodownloader import FixedYahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split

# ── Constants ──
SEEDS = [42, 123, 456]
TICKERS = ["AAPL", "AMZN", "CRM", "MSFT", "NFLX"]
LLM_CONFIGS = ["gemini", "qwen_base", "qwen_qlora", "qwen_qalora", "no_nlp"]
TRAIN_START = "2022-04-01"
TRAIN_END = "2024-07-31"
TEST_START = "2024-08-01"
TEST_END = "2025-02-28"

TECH_INDICATORS = ["macd", "boll_ub", "boll_lb", "rsi_30", "cci_30", "dx_30", "close_30_sma", "close_60_sma"]
FUNDAMENTAL_INDICATORS = [
    "news_relevance", "sentiment", "price_impact_potential",
    "trend_direction", "earnings_impact", "investor_confidence",
    "risk_profile_change",
]

COLUMN_MAPPING = {
    "News Relevance": "news_relevance",
    "Sentiment": "sentiment",
    "Price Impact Potential": "price_impact_potential",
    "Trend Direction": "trend_direction",
    "Earnings Impact": "earnings_impact",
    "Investor Confidence": "investor_confidence",
    "Risk Profile Change": "risk_profile_change",
}

# ── PPO Config (stable, reference-aligned) ──
TRANSACTION_COST = 0.001  # 0.1%
TOTAL_TIMESTEPS = 200000
REWARD_TYPE = "dollar_delta"
REWARD_SCALING = 1
PPO_PARAMS = dict(
    n_steps=2048,
    ent_coef=0.01,
    learning_rate=0.00025,
    batch_size=128,
)

# ── Output ──
OUTPUT_DIR = OUR_CODE / "llm_data_comparison_results"


# ============================================================================
# DATA LOADING
# ============================================================================

def load_nlp_data(llm_config, ticker):
    """Load NLP signal data for a given LLM config and ticker."""
    if llm_config == "no_nlp":
        return None

    if llm_config == "gemini":
        path = OUR_CODE / "ppo_data" / f"gemini_ppo_{ticker}_data.csv"
    else:
        path = OUR_CODE / "LLM_data" / f"{llm_config}_ppo_{ticker}_data.csv"

    if not path.exists():
        raise FileNotFoundError(f"NLP data not found: {path}")

    df = pd.read_csv(path)

    # Rename columns
    if "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})

    # Map NLP feature columns (Title Case -> snake_case)
    for old_name, new_name in COLUMN_MAPPING.items():
        if old_name in df.columns:
            df = df.rename(columns={old_name: new_name})

    # Drop non-feature columns
    drop_cols = ["Adj Close Price", "Returns", "Bin Label", "Prompt", "ticker",
                 "Parse Success", "Parse Error", "Raw Response", "close_nlp"]
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Ensure date is string
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    return df


def load_price_data(ticker):
    """Load price data using FixedYahooDownloader + FeatureEngineer."""
    df = FixedYahooDownloader(
        start_date=TRAIN_START,
        end_date=TEST_END,
        ticker_list=[ticker]
    ).fetch_data()

    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=TECH_INDICATORS,
        use_vix=True,
        use_turbulence=True,
        user_defined_feature=False,
    )
    processed = fe.preprocess_data(df)
    processed = processed.ffill().bfill().fillna(0)
    processed = processed.sort_values(["date", "tic"], ignore_index=True)
    return processed


def prepare_combined_data(price_df, nlp_df, llm_config, ticker):
    """Merge price + NLP data, apply shift(1), split train/test."""
    if llm_config == "no_nlp":
        result = price_df.copy()
        for col in FUNDAMENTAL_INDICATORS:
            result[col] = 0.0
    else:
        result = price_df.merge(nlp_df, on="date", how="left")

        # Handle duplicate columns from merge
        for col in result.columns:
            if col.endswith("_x"):
                base = col[:-2]
                if base + "_y" in result.columns:
                    result[base] = result[col]
                    result = result.drop(columns=[col, base + "_y"])
            elif col.endswith("_y") and col[:-2] + "_x" not in result.columns:
                result = result.rename(columns={col: col[:-2]})

        # Drop leftover columns
        drop_extra = [c for c in result.columns if c.endswith("_nlp")]
        result = result.drop(columns=drop_extra, errors="ignore")

        # Ensure all NLP columns exist
        for col in FUNDAMENTAL_INDICATORS:
            if col not in result.columns:
                result[col] = 0.0

        # shift(1) NLP features to prevent look-ahead
        for col in FUNDAMENTAL_INDICATORS:
            result[col] = result.groupby("tic")[col].shift(1)

        result = result.fillna(0)

    # Train/test split
    train = data_split(result, TRAIN_START, TRAIN_END)
    test = data_split(result, TEST_START, TEST_END)

    return train, test, result


# ============================================================================
# NORMALIZED ENVIRONMENT WRAPPER
# ============================================================================

class NormalizedObsWrapper(gym.Wrapper):
    """
    Wraps StockTradingEnv to selectively upscale NLP features so they are
    visible to the policy network alongside cash/price/tech indicators.

    Problem: Raw state vector has extreme scale differences:
      - cash: ~100,000
      - close price: ~200
      - tech indicators: various (MACD ~-10 to +10, RSI ~0-100, etc.)
      - NLP features: [-2, 2]

    Previous experiments showed NLP features contributed only 0.8% to policy
    decisions because they are invisible to gradients.

    Solution: Instead of z-score normalizing the ENTIRE observation (which
    destabilizes PPO by changing the scale it has already learned), we
    selectively UPSACLE NLP features by a constant factor so they are in
    the same order of magnitude as the tech indicators (~0-100 range).

    Specifically, we multiply each NLP feature by a scaling factor that
    maps its [-2, 2] range to approximately [-50, 50], matching tech indicator
    scales. We also z-score normalize cash and price to ~[-3, 3] range.

    State layout (single stock, stock_dim=1):
      [0]     cash           ~100,000  -> normalize to ~[-3,3]
      [1]     close price    ~200      -> normalize to ~[-3,3]
      [2]     stock shares   ~0-1000   -> normalize to ~[-3,3]
      [3:11]  8 tech indicators  (keep as-is, already reasonable scale)
      [11:18] 7 NLP features     [-2,2] -> upscale by 25x to [-50,50]
    """

    def __init__(self, env, obs_mean=None, obs_std=None, nlp_scale=25.0, eps=1e-8):
        super().__init__(env)
        self.eps = eps
        self.nlp_scale = nlp_scale
        self.n_features = env.state_space  # total observation dim

        # Determine which indices are NLP features
        stock_dim = env.stock_dim
        n_tech = len(env.tech_indicator_list) * stock_dim
        n_nlp = len(env.fundamental_indicator_list) * stock_dim
        self.nlp_start = 1 + 2 * stock_dim + n_tech  # index where NLP features begin
        self.nlp_end = self.nlp_start + n_nlp

        # Cash/price/holdings normalization stats
        if obs_mean is not None and obs_std is not None:
            self.obs_mean = np.array(obs_mean, dtype=np.float32)
            self.obs_std = np.array(obs_std, dtype=np.float32)
        else:
            self.obs_mean = None
            self.obs_std = None

    def compute_stats_from_data(self, train_df):
        """Compute normalization statistics for cash/price/holdings only."""
        from finrl.meta.env_primo_trading.env_primorl import StockTradingEnv

        stock_dim = 1
        state_space = 1 + 2 * stock_dim + len(TECH_INDICATORS) * stock_dim + len(FUNDAMENTAL_INDICATORS) * stock_dim

        env_kwargs = dict(
            hmax=1000, initial_amount=100000,
            buy_cost_pct=[TRANSACTION_COST], sell_cost_pct=[TRANSACTION_COST],
            num_stock_shares=[0], stock_dim=stock_dim,
            state_space=state_space, action_space=stock_dim,
            tech_indicator_list=TECH_INDICATORS,
            fundamental_indicator_list=FUNDAMENTAL_INDICATORS,
            reward_type=REWARD_TYPE, reward_scaling=REWARD_SCALING,
            initial=True, print_verbosity=999999,
        )

        temp_env = StockTradingEnv(df=train_df, **env_kwargs)
        obs, _ = temp_env.reset()

        all_obs = [np.array(obs, dtype=np.float64)]
        done = False
        while not done:
            action = temp_env.action_space.sample()
            obs, reward, terminated, truncated, info = temp_env.step(action)
            all_obs.append(np.array(obs, dtype=np.float64))
            done = terminated or truncated

        all_obs = np.array(all_obs)
        self.obs_mean = np.mean(all_obs, axis=0).astype(np.float32)
        self.obs_std = np.std(all_obs, axis=0).astype(np.float32)
        self.obs_std = np.maximum(self.obs_std, self.eps)

        return self.obs_mean, self.obs_std

    def _transform(self, obs):
        """Apply selective normalization: z-score cash/price/holdings, upscale NLP."""
        obs = np.array(obs, dtype=np.float32)

        if self.obs_mean is not None:
            # Z-score normalize first 3 features (cash, price, shares)
            for i in range(min(3, len(obs))):
                obs[i] = (obs[i] - self.obs_mean[i]) / self.obs_std[i]

        # Upscale NLP features
        obs[self.nlp_start:self.nlp_end] *= self.nlp_scale

        return obs

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            obs, info = result
            return self._transform(obs), info
        return self._transform(result)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._transform(obs), reward, terminated, truncated, info


# ============================================================================
# TRAINING & EVALUATION
# ============================================================================

def train_and_eval(llm_config, ticker, seed, output_dir):
    """Train PPO with given LLM config, ticker, seed. Return metrics dict."""
    from finrl.meta.env_primo_trading.env_primorl import StockTradingEnv
    from finrl.agents.stablebaselines3.models import DRLAgent
    from stable_baselines3.common.logger import configure

    run_name = f"{llm_config}_{ticker}_seed{seed}"
    result_dir = output_dir / run_name
    result_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Training: {run_name}")
    print(f"  LLM data: {llm_config} | Ticker: {ticker} | Seed: {seed}")
    print(f"  Reward: {REWARD_TYPE} sc={REWARD_SCALING} | Timesteps: {TOTAL_TIMESTEPS}")
    print(f"  Observation normalization: ENABLED (selective: z-score cash/price, 25x NLP upscale)")
    print(f"{'='*60}")

    try:
        # Load data
        nlp_df = load_nlp_data(llm_config, ticker)
        price_df = load_price_data(ticker)
        train_df, test_df, full_df = prepare_combined_data(price_df, nlp_df, llm_config, ticker)

        # Set up environment
        stock_dim = 1
        state_space = 1 + 2 * stock_dim + len(TECH_INDICATORS) * stock_dim + len(FUNDAMENTAL_INDICATORS) * stock_dim

        env_kwargs = dict(
            hmax=1000,
            initial_amount=100000,
            buy_cost_pct=[TRANSACTION_COST],
            sell_cost_pct=[TRANSACTION_COST],
            num_stock_shares=[0],
            stock_dim=stock_dim,
            state_space=state_space,
            action_space=stock_dim,
            tech_indicator_list=TECH_INDICATORS,
            fundamental_indicator_list=FUNDAMENTAL_INDICATORS,
            reward_type=REWARD_TYPE,
            reward_scaling=REWARD_SCALING,
            initial=True,
            print_verbosity=999999,
        )

        # Compute normalization stats from a temporary env
        e_temp = StockTradingEnv(df=train_df, **env_kwargs)
        norm_temp = NormalizedObsWrapper(e_temp)
        obs_mean, obs_std = norm_temp.compute_stats_from_data(train_df)

        # Save normalization stats
        np.savez(result_dir / "norm_stats.npz", mean=obs_mean, std=obs_std)

        # Create factory function for DummyVecEnv (avoids closure issues)
        def make_train_env():
            raw = StockTradingEnv(df=train_df, **env_kwargs)
            return NormalizedObsWrapper(raw, obs_mean=obs_mean, obs_std=obs_std)

        # Wrap in DummyVecEnv for SB3 compatibility
        from stable_baselines3.common.vec_env import DummyVecEnv
        train_env = DummyVecEnv([make_train_env])

        # Train PPO
        from stable_baselines3 import PPO
        import torch as th

        model = PPO(
            "MlpPolicy",
            train_env,
            n_steps=PPO_PARAMS["n_steps"],
            ent_coef=PPO_PARAMS["ent_coef"],
            learning_rate=PPO_PARAMS["learning_rate"],
            batch_size=PPO_PARAMS["batch_size"],
            seed=seed,
            verbose=0,
            device="auto",
        )

        t0 = time.time()
        model.learn(total_timesteps=TOTAL_TIMESTEPS)
        train_time = time.time() - t0
        print(f"  Training time: {train_time:.0f}s")

        # Evaluate on test data
        e_test_raw = StockTradingEnv(df=test_df, **env_kwargs)
        e_test = NormalizedObsWrapper(e_test_raw, obs_mean=obs_mean, obs_std=obs_std)

        obs, _ = e_test.reset()
        account_values = [e_test.unwrapped.asset_memory[-1]]
        actions_list = []

        for _ in range(len(test_df.index.unique())):
            action, _ = model.predict(obs, deterministic=True)
            action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
            action = np.clip(action, -1.0, 1.0)
            obs, reward, terminated, truncated, info = e_test.step(action)
            account_values.append(e_test.unwrapped.asset_memory[-1])
            actions_list.append(action)
            if terminated or truncated:
                break

        # Compute metrics
        metrics = compute_metrics(account_values)
        trade_metrics = compute_trade_metrics(actions_list)
        metrics.update(trade_metrics)
        metrics["train_time"] = train_time
        metrics["llm_config"] = llm_config
        metrics["ticker"] = ticker
        metrics["seed"] = seed
        metrics["reward_type"] = REWARD_TYPE
        metrics["reward_scaling"] = REWARD_SCALING
        metrics["obs_normalized"] = True

        # Save results
        pd.DataFrame({"account_value": account_values}).to_csv(
            result_dir / "account_value.csv", index=False
        )
        with open(result_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        print(f"  Return: {metrics['total_return']*100:+.2f}% | Sharpe: {metrics['sharpe_ratio']:.2f} | Trades: {metrics['num_trades']}")
        return metrics

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {
            "llm_config": llm_config, "ticker": ticker, "seed": seed,
            "error": str(e), "total_return": 0, "sharpe_ratio": 0,
            "num_trades": 0, "do_nothing_pct": 100,
        }


def compute_metrics(account_values, risk_free_rate=0.03):
    """Compute trading metrics from account value series."""
    values = np.array(account_values)
    n = len(values)
    initial = values[0]
    final = values[-1]
    total_return = (final - initial) / initial

    daily_returns = np.diff(values) / values[:-1]
    daily_returns = daily_returns[~np.isnan(daily_returns)]

    if len(daily_returns) == 0:
        return {
            "total_return": 0, "annualized_return": 0, "sharpe_ratio": 0,
            "max_drawdown": 0, "max_drawdown_duration": 0, "volatility": 0,
            "final_value": final,
        }

    mean_ret = np.mean(daily_returns)
    std_ret = np.std(daily_returns) + 1e-10
    daily_rf = risk_free_rate / 252
    sharpe = (mean_ret - daily_rf) / std_ret * np.sqrt(252)

    annualized_return = (1 + total_return) ** (252 / max(n - 1, 1)) - 1
    volatility = np.std(daily_returns) * np.sqrt(252)

    # Max drawdown
    peak = values[0]
    max_dd = 0
    max_dd_duration = 0
    dd_start = 0
    for i in range(n):
        if values[i] > peak:
            peak = values[i]
            dd_start = i
        dd = (peak - values[i]) / peak
        if dd > max_dd:
            max_dd = dd
            max_dd_duration = i - dd_start

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "max_drawdown_duration": max_dd_duration,
        "volatility": volatility,
        "final_value": final,
    }


def compute_trade_metrics(actions_list):
    """Compute trade-level metrics from actions list."""
    changes = 0
    prev_action = 0
    for action in actions_list:
        a = action[0] if isinstance(action, np.ndarray) else float(action)
        if a != prev_action:
            changes += 1
        prev_action = a

    total_days = len(actions_list)
    do_nothing_days = sum(
        1 for a in actions_list
        if (a[0] if isinstance(a, np.ndarray) else a) == 0
    )
    do_nothing_pct = do_nothing_days / total_days * 100 if total_days > 0 else 0

    return {"num_trades": changes, "do_nothing_pct": do_nothing_pct}


# ============================================================================
# MAIN
# ============================================================================

def run_all():
    """Run all experiments: 5 LLM configs x 5 tickers x 5 seeds = 125 runs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    total = len(LLM_CONFIGS) * len(TICKERS) * len(SEEDS)
    done = 0

    for llm_config in LLM_CONFIGS:
        for ticker in TICKERS:
            for seed in SEEDS:
                done += 1
                print(f"\n[{done}/{total}] ", end="")
                m = train_and_eval(llm_config, ticker, seed, OUTPUT_DIR)
                all_metrics.append(m)

    # Save all results
    results_df = pd.DataFrame(all_metrics)
    results_df.to_csv(OUTPUT_DIR / "all_results.csv", index=False)

    # Generate summary
    summary = generate_summary(all_metrics)
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Print summary
    print_summary(summary)

    return all_metrics, summary


def generate_summary(all_metrics):
    """Generate structured summary."""
    summary = {
        "experiment_info": {
            "timestamp": datetime.now().isoformat(),
            "algorithm": "PPO",
            "reward_type": REWARD_TYPE,
            "reward_scaling": REWARD_SCALING,
            "timesteps": TOTAL_TIMESTEPS,
            "seeds": SEEDS,
            "tickers": TICKERS,
            "llm_configs": LLM_CONFIGS,
            "transaction_cost": TRANSACTION_COST,
            "train_period": f"{TRAIN_START} to {TRAIN_END}",
            "test_period": f"{TEST_START} to {TEST_END}",
            "nlp_shift": 1,
            "downloader": "FixedYahooDownloader",
            "obs_normalized": True,
            "normalization_method": "selective: z-score cash/price/holdings, 25x upscale NLP features",
        },
        "per_config_ticker": {},
        "per_config_overall": {},
    }

    for llm_config in LLM_CONFIGS:
        summary["per_config_ticker"][llm_config] = {}
        for ticker in TICKERS:
            runs = [m for m in all_metrics
                    if m.get("llm_config") == llm_config and m.get("ticker") == ticker]
            if not runs:
                continue

            active = [r for r in runs if r.get("total_return", 0) != 0]

            summary["per_config_ticker"][llm_config][ticker] = {
                "n_runs": len(runs),
                "n_active": len(active),
                "mean_return": np.mean([r["total_return"] for r in runs]) * 100,
                "mean_return_active": np.mean([r["total_return"] for r in active]) * 100 if active else 0,
                "mean_sharpe_active": np.mean([r["sharpe_ratio"] for r in active]) if active else 0,
                "std_return": np.std([r["total_return"] for r in runs]) * 100,
                "mean_trades": np.mean([r.get("num_trades", 0) for r in runs]),
            }

    for llm_config in LLM_CONFIGS:
        runs = [m for m in all_metrics if m.get("llm_config") == llm_config]
        active = [r for r in runs if r.get("total_return", 0) != 0]

        summary["per_config_overall"][llm_config] = {
            "n_runs": len(runs),
            "n_active": len(active),
            "mean_return": np.mean([r["total_return"] for r in runs]) * 100,
            "mean_return_active": np.mean([r["total_return"] for r in active]) * 100 if active else 0,
            "mean_sharpe_active": np.mean([r["sharpe_ratio"] for r in active]) if active else 0,
            "std_return": np.std([r["total_return"] for r in runs]) * 100,
            "mean_trades": np.mean([r.get("num_trades", 0) for r in runs]),
        }

    return summary


def print_summary(summary):
    """Print formatted summary."""
    print("\n" + "=" * 80)
    print("LLM DATA IMPACT TEST — RESULTS SUMMARY")
    print("=" * 80)

    info = summary["experiment_info"]
    print(f"  Algorithm: {info['algorithm']}, Reward: {info['reward_type']} sc={info['reward_scaling']}")
    print(f"  Obs normalization: {info['obs_normalized']} ({info['normalization_method']})")
    print(f"  Seeds: {info['seeds']}")
    print(f"  Tickers: {info['tickers']}")
    print()

    # Overall comparison
    print(f"{'Config':15s} {'N':>3s} {'Active':>6s} {'Return':>9s} {'Ret(act)':>9s} {'Sharpe':>7s} {'Trades':>7s}")
    print("-" * 60)
    for cfg in LLM_CONFIGS:
        d = summary["per_config_overall"][cfg]
        print(f"  {cfg:15s} {d['n_runs']:3d} {d['n_active']:5d}   {d['mean_return']:+7.2f}%  {d['mean_return_active']:+7.2f}%  {d['mean_sharpe_active']:5.2f}  {d['mean_trades']:5.1f}")

    print()
    print("Per-ticker active returns:")
    for ticker in TICKERS:
        print(f"  {ticker}:", end="")
        for cfg in LLM_CONFIGS:
            d = summary["per_config_ticker"][cfg].get(ticker, {})
            ret = d.get("mean_return_active", 0)
            print(f"  {cfg[:5]}={ret:+.1f}%", end="")
        print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Single LLM config to run")
    parser.add_argument("--ticker", type=str, default=None, help="Single ticker to run")
    parser.add_argument("--seed", type=int, default=None, help="Single seed to run")
    args = parser.parse_args()

    if args.config or args.ticker or args.seed:
        # Run single experiment
        configs = [args.config] if args.config else LLM_CONFIGS
        tickers = [args.ticker] if args.ticker else TICKERS
        seeds = [args.seed] if args.seed else SEEDS

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        all_metrics = []
        for cfg in configs:
            for t in tickers:
                for s in seeds:
                    m = train_and_eval(cfg, t, s, OUTPUT_DIR)
                    all_metrics.append(m)

        results_df = pd.DataFrame(all_metrics)
        results_df.to_csv(OUTPUT_DIR / "all_results.csv", index=False)
        summary = generate_summary(all_metrics)
        with open(OUTPUT_DIR / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print_summary(summary)
    else:
        run_all()
