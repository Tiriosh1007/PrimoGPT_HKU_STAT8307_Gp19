#!/usr/bin/env python3
"""
LLM Data Impact Test v4 — Ensemble of Active Agents
====================================================
Purpose: Compare trading performance under different LLM data sources,
         using an ENSEMBLE of active agents to reduce seed variance.

Design:
  Phase 1: Train 10 seeds per (LLM config, ticker) = 250 runs
  Phase 2: For each (LLM config, ticker):
           - Identify "active" agents (those that actually traded during training eval)
           - Ensemble their actions: mean(action_i) across active agents
           - Evaluate the ensemble policy on test data
  Phase 3: Compare ensemble performance across LLM configs

The ensemble approach reduces the dominant seed variance and gives a
cleaner signal on whether LLM data quality affects trading.

5 LLM data sources:
  1. Gemini 3.1 Pro
  2. Qwen3.5-27B base
  3. Qwen3.5-27B + QLoRA
  4. Qwen3.5-27B + QA-LoRA
  5. No-NLP (tech indicators only — ablation baseline)
"""

import pandas as pd
import numpy as np
import gymnasium as gym
import os, sys, json, time, warnings, pickle
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
SEEDS = [42, 123, 456, 789, 2024, 314, 271, 1618, 999, 7]
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

# ── PPO Config ──
TRANSACTION_COST = 0.001  # 0.1%
TOTAL_TIMESTEPS = 200000
REWARD_TYPE = "diff_sharpe"
REWARD_SCALING = 1
NLP_SCALE = 25.0
ACTIVE_THRESHOLD = 0.001  # 0.1% return threshold to count as "active"

# ── Output ──
OUTPUT_DIR = OUR_CODE / "llm_ensemble_diff_sharpe_results"


# ============================================================================
# DATA LOADING (same as v3)
# ============================================================================

def load_nlp_data(llm_config, ticker):
    if llm_config == "no_nlp":
        return None
    if llm_config == "gemini":
        path = OUR_CODE / "ppo_data" / f"gemini_ppo_{ticker}_data.csv"
    else:
        path = OUR_CODE / "LLM_data" / f"{llm_config}_ppo_{ticker}_data.csv"
    if not path.exists():
        raise FileNotFoundError(f"NLP data not found: {path}")
    df = pd.read_csv(path)
    if "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})
    for old_name, new_name in COLUMN_MAPPING.items():
        if old_name in df.columns:
            df = df.rename(columns={old_name: new_name})
    drop_cols = ["Adj Close Price", "Returns", "Bin Label", "Prompt", "ticker",
                 "Parse Success", "Parse Error", "Raw Response", "close_nlp"]
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df


def load_price_data(ticker):
    df = FixedYahooDownloader(
        start_date=TRAIN_START, end_date=TEST_END, ticker_list=[ticker]
    ).fetch_data()
    fe = FeatureEngineer(
        use_technical_indicator=True, tech_indicator_list=TECH_INDICATORS,
        use_vix=True, use_turbulence=True, user_defined_feature=False,
    )
    processed = fe.preprocess_data(df)
    processed = processed.ffill().bfill().fillna(0)
    processed = processed.sort_values(["date", "tic"], ignore_index=True)
    return processed


def prepare_combined_data(price_df, nlp_df, llm_config, ticker):
    if llm_config == "no_nlp":
        result = price_df.copy()
        for col in FUNDAMENTAL_INDICATORS:
            result[col] = 0.0
    else:
        result = price_df.merge(nlp_df, on="date", how="left")
        for col in result.columns:
            if col.endswith("_x"):
                base = col[:-2]
                if base + "_y" in result.columns:
                    result[base] = result[col]
                    result = result.drop(columns=[col, base + "_y"])
        drop_extra = [c for c in result.columns if c.endswith("_nlp")]
        result = result.drop(columns=drop_extra, errors="ignore")
        for col in FUNDAMENTAL_INDICATORS:
            if col not in result.columns:
                result[col] = 0.0
        for col in FUNDAMENTAL_INDICATORS:
            result[col] = result.groupby("tic")[col].shift(1)
        result = result.fillna(0)

    train = data_split(result, TRAIN_START, TRAIN_END)
    test = data_split(result, TEST_START, TEST_END)
    return train, test, result


# ============================================================================
# NLP-ONLY UPSCALE WRAPPER (for diff_sharpe reward)
# ============================================================================

class NormalizedObsWrapper(gym.Wrapper):
    """
    Upscales NLP features by nlp_scale factor only. No z-score normalization.
    This works with diff_sharpe reward where z-score normalization causes
    do-nothing policies.
    """
    def __init__(self, env, obs_mean=None, obs_std=None, nlp_scale=25.0, eps=1e-8):
        super().__init__(env)
        self.nlp_scale = nlp_scale
        stock_dim = env.stock_dim
        n_tech = len(env.tech_indicator_list) * stock_dim
        n_nlp = len(env.fundamental_indicator_list) * stock_dim
        self.nlp_start = 1 + 2 * stock_dim + n_tech
        self.nlp_end = self.nlp_start + n_nlp
        # obs_mean/std accepted but NOT used (kept for API compatibility)
        self.obs_mean = None
        self.obs_std = None

    def compute_stats_from_data(self, train_df):
        """No-op for NLP-only upscale, but returns dummy arrays for API compat."""
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
        # Create temp env to get observation dimension
        temp_env = StockTradingEnv(df=train_df, **env_kwargs)
        obs, _ = temp_env.reset()
        dim = len(obs)
        self.obs_mean = np.zeros(dim, dtype=np.float32)
        self.obs_std = np.ones(dim, dtype=np.float32)
        return self.obs_mean, self.obs_std

    def _transform(self, obs):
        obs = np.array(obs, dtype=np.float32)
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
# PHASE 1: TRAINING
# ============================================================================

def get_env_kwargs():
    stock_dim = 1
    state_space = 1 + 2 * stock_dim + len(TECH_INDICATORS) * stock_dim + len(FUNDAMENTAL_INDICATORS) * stock_dim
    return dict(
        hmax=1000, initial_amount=100000,
        buy_cost_pct=[TRANSACTION_COST], sell_cost_pct=[TRANSACTION_COST],
        num_stock_shares=[0], stock_dim=stock_dim,
        state_space=state_space, action_space=stock_dim,
        tech_indicator_list=TECH_INDICATORS,
        fundamental_indicator_list=FUNDAMENTAL_INDICATORS,
        reward_type=REWARD_TYPE, reward_scaling=REWARD_SCALING,
        initial=True, print_verbosity=999999,
    )


def train_single(llm_config, ticker, seed, output_dir):
    """Train a single PPO agent. Save model + metadata. Return (is_active, metrics)."""
    from finrl.meta.env_primo_trading.env_primorl import StockTradingEnv
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    run_name = f"{llm_config}_{ticker}_seed{seed}"
    result_dir = output_dir / "models" / run_name
    result_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already done
    if (result_dir / "metrics.json").exists():
        with open(result_dir / "metrics.json") as f:
            metrics = json.load(f)
        is_active = abs(metrics.get("total_return", 0)) > ACTIVE_THRESHOLD
        return is_active, metrics

    try:
        nlp_df = load_nlp_data(llm_config, ticker)
        price_df = load_price_data(ticker)
        train_df, test_df, full_df = prepare_combined_data(price_df, nlp_df, llm_config, ticker)

        env_kwargs = get_env_kwargs()

        # Compute normalization stats
        e_temp = StockTradingEnv(df=train_df, **env_kwargs)
        norm_temp = NormalizedObsWrapper(e_temp)
        obs_mean, obs_std = norm_temp.compute_stats_from_data(train_df)
        np.savez(result_dir / "norm_stats.npz", mean=obs_mean, std=obs_std)

        # Train
        def make_train_env():
            raw = StockTradingEnv(df=train_df, **env_kwargs)
            return NormalizedObsWrapper(raw, obs_mean=obs_mean, obs_std=obs_std)

        train_env = DummyVecEnv([make_train_env])

        model = PPO(
            "MlpPolicy", train_env,
            n_steps=2048, ent_coef=0.01, learning_rate=0.00025,
            batch_size=128, seed=seed, verbose=0, device="auto",
        )
        t0 = time.time()
        model.learn(total_timesteps=TOTAL_TIMESTEPS)
        train_time = time.time() - t0

        # Quick eval to determine if active (on training data last episode)
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

        metrics = compute_metrics(account_values)
        trade_metrics = compute_trade_metrics(actions_list)
        metrics.update(trade_metrics)
        metrics["train_time"] = train_time
        metrics["llm_config"] = llm_config
        metrics["ticker"] = ticker
        metrics["seed"] = seed

        # Save model and metadata
        model.save(str(result_dir / "model"))
        with open(result_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        pd.DataFrame({"account_value": account_values}).to_csv(
            result_dir / "account_value.csv", index=False
        )

        is_active = abs(metrics.get("total_return", 0)) > ACTIVE_THRESHOLD

        ret_str = f"{metrics['total_return']*100:+.2f}%"
        print(f"  {run_name}: {ret_str} | Sharpe {metrics['sharpe_ratio']:.2f} | {'ACTIVE' if is_active else 'DO-NOTHING'}")

        return is_active, metrics

    except Exception as e:
        print(f"  {run_name}: ERROR - {e}")
        import traceback
        traceback.print_exc()
        return False, {"llm_config": llm_config, "ticker": ticker, "seed": seed, "error": str(e),
                       "total_return": 0, "sharpe_ratio": 0, "num_trades": 0}


# ============================================================================
# PHASE 2: ENSEMBLE EVALUATION
# ============================================================================

def evaluate_ensemble(llm_config, ticker, output_dir):
    """
    Load all trained models for (llm_config, ticker), filter active ones,
    then evaluate using ensemble (mean action).
    Returns ensemble metrics dict.
    """
    from finrl.meta.env_primo_trading.env_primorl import StockTradingEnv
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    print(f"\n  Ensemble: {llm_config} / {ticker}")

    # Collect all active models
    active_models = []
    obs_mean = None
    obs_std = None

    for seed in SEEDS:
        run_name = f"{llm_config}_{ticker}_seed{seed}"
        model_dir = output_dir / "models" / run_name
        metrics_path = model_dir / "metrics.json"

        if not metrics_path.exists():
            continue

        with open(metrics_path) as f:
            metrics = json.load(f)

        # Check if active
        if abs(metrics.get("total_return", 0)) <= ACTIVE_THRESHOLD:
            print(f"    Skip seed {seed}: do-nothing ({metrics.get('total_return',0)*100:+.2f}%)")
            continue

        # Load model
        model_path = model_dir / "model"
        if not Path(str(model_path) + ".zip").exists():
            continue

        model = PPO.load(str(model_path))

        # Load normalization stats (same for all seeds of same config/ticker)
        if obs_mean is None:
            stats = np.load(model_dir / "norm_stats.npz")
            obs_mean = stats["mean"]
            obs_std = stats["std"]

        active_models.append((seed, model))

    n_active = len(active_models)
    print(f"    Active models: {n_active}/{len(SEEDS)}")

    if n_active == 0:
        return {
            "llm_config": llm_config, "ticker": ticker,
            "n_active": 0, "total_return": 0, "sharpe_ratio": 0,
            "num_trades": 0, "error": "No active models",
        }

    # Load test data
    nlp_df = load_nlp_data(llm_config, ticker)
    price_df = load_price_data(ticker)
    train_df, test_df, full_df = prepare_combined_data(price_df, nlp_df, llm_config, ticker)

    env_kwargs = get_env_kwargs()

    # Create test env
    e_test_raw = StockTradingEnv(df=test_df, **env_kwargs)
    e_test = NormalizedObsWrapper(e_test_raw, obs_mean=obs_mean, obs_std=obs_std)

    obs, _ = e_test.reset()
    account_values = [e_test.unwrapped.asset_memory[-1]]
    ensemble_actions = []

    n_steps = len(test_df.index.unique())
    for step in range(n_steps):
        # Get action from each active model
        actions = []
        for seed, model in active_models:
            action, _ = model.predict(obs, deterministic=True)
            action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
            action = np.clip(action, -1.0, 1.0)
            actions.append(action)

        # Ensemble: mean action
        ensemble_action = np.mean(actions, axis=0)
        ensemble_actions.append(ensemble_action)

        # Step the env with ensemble action
        obs, reward, terminated, truncated, info = e_test.step(ensemble_action)
        account_values.append(e_test.unwrapped.asset_memory[-1])
        if terminated or truncated:
            break

    # Compute metrics
    metrics = compute_metrics(account_values)
    trade_metrics = compute_trade_metrics(ensemble_actions)
    metrics.update(trade_metrics)
    metrics["llm_config"] = llm_config
    metrics["ticker"] = ticker
    metrics["n_active"] = n_active
    metrics["n_total_seeds"] = len(SEEDS)

    # Save ensemble results
    ens_dir = output_dir / "ensemble" / f"{llm_config}_{ticker}"
    ens_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"account_value": account_values}).to_csv(ens_dir / "account_value.csv", index=False)
    with open(ens_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    ret_str = f"{metrics['total_return']*100:+.2f}%"
    print(f"    Result: {ret_str} | Sharpe {metrics['sharpe_ratio']:.2f} | Trades {metrics['num_trades']}")

    return metrics


# ============================================================================
# METRICS
# ============================================================================

def compute_metrics(account_values, risk_free_rate=0.03):
    values = np.array(account_values)
    n = len(values)
    initial = values[0]
    final = values[-1]
    total_return = (final - initial) / initial

    daily_returns = np.diff(values) / values[:-1]
    daily_returns = daily_returns[~np.isnan(daily_returns)]

    if len(daily_returns) == 0:
        return {"total_return": 0, "annualized_return": 0, "sharpe_ratio": 0,
                "max_drawdown": 0, "max_drawdown_duration": 0, "volatility": 0, "final_value": final}

    mean_ret = np.mean(daily_returns)
    std_ret = np.std(daily_returns) + 1e-10
    daily_rf = risk_free_rate / 252
    sharpe = (mean_ret - daily_rf) / std_ret * np.sqrt(252)
    annualized_return = (1 + total_return) ** (252 / max(n - 1, 1)) - 1
    volatility = np.std(daily_returns) * np.sqrt(252)

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
        if abs(a[0] if isinstance(a, np.ndarray) else a) < 0.01
    )
    do_nothing_pct = do_nothing_days / total_days * 100 if total_days > 0 else 0
    return {"num_trades": changes, "do_nothing_pct": do_nothing_pct}


# ============================================================================
# MAIN
# ============================================================================

def run_phase1(output_dir):
    """Phase 1: Train all 250 models."""
    print("=" * 70)
    print("PHASE 1: TRAINING (250 runs)")
    print("=" * 70)

    all_metrics = []
    total = len(LLM_CONFIGS) * len(TICKERS) * len(SEEDS)
    done = 0

    for llm_config in LLM_CONFIGS:
        for ticker in TICKERS:
            for seed in SEEDS:
                done += 1
                print(f"\n[{done}/{total}] ", end="")
                is_active, metrics = train_single(llm_config, ticker, seed, output_dir)
                metrics["is_active"] = is_active
                all_metrics.append(metrics)

    results_df = pd.DataFrame(all_metrics)
    results_df.to_csv(output_dir / "all_training_results.csv", index=False)
    return all_metrics


def run_phase2(output_dir):
    """Phase 2: Ensemble evaluation."""
    print("\n" + "=" * 70)
    print("PHASE 2: ENSEMBLE EVALUATION")
    print("=" * 70)

    all_ensemble_metrics = []
    for llm_config in LLM_CONFIGS:
        for ticker in TICKERS:
            m = evaluate_ensemble(llm_config, ticker, output_dir)
            all_ensemble_metrics.append(m)

    results_df = pd.DataFrame(all_ensemble_metrics)
    results_df.to_csv(output_dir / "ensemble_results.csv", index=False)
    return all_ensemble_metrics


def generate_report(training_metrics, ensemble_metrics, output_dir):
    """Generate final comparison report."""
    from scipy import stats

    print("\n" + "=" * 80)
    print("FINAL REPORT: LLM DATA IMPACT — ENSEMBLE OF ACTIVE AGENTS")
    print("=" * 80)

    # ── Training summary ──
    train_df = pd.DataFrame(training_metrics)
    print(f"\n--- Training Summary ---")
    print(f"Total runs: {len(train_df)}")

    for cfg in LLM_CONFIGS:
        sub = train_df[train_df['llm_config'] == cfg]
        active = sub[sub['is_active'] == True]
        print(f"  {cfg:15s}: {len(active)}/{len(sub)} active ({len(active)/len(sub)*100:.0f}%)")

    # ── Ensemble results ──
    ens_df = pd.DataFrame(ensemble_metrics)
    print(f"\n--- Ensemble Results ---")
    print(f"{'Config':15s} {'AAPL':>8s} {'AMZN':>8s} {'CRM':>8s} {'MSFT':>8s} {'NFLX':>8s} {'MEAN':>8s} {'Sharpe':>8s}")
    print("-" * 80)

    for cfg in LLM_CONFIGS:
        sub = ens_df[ens_df['llm_config'] == cfg]
        rets = []
        sharpes = []
        row = f"  {cfg:13s}"
        for ticker in TICKERS:
            t = sub[sub['ticker'] == ticker]
            if len(t) > 0:
                ret = t.iloc[0]['total_return'] * 100
                sh = t.iloc[0]['sharpe_ratio']
                row += f" {ret:+7.1f}%"
                rets.append(t.iloc[0]['total_return'])
                sharpes.append(sh)
            else:
                row += f"    N/A "
        if rets:
            row += f" {np.mean(rets)*100:+7.1f}%"
            row += f" {np.mean(sharpes):7.2f}"
        print(row)

    # ── Statistical test on ensemble returns ──
    groups = []
    for cfg in LLM_CONFIGS:
        sub = ens_df[ens_df['llm_config'] == cfg]
        groups.append(sub['total_return'].values)

    try:
        f_stat, p_val = stats.f_oneway(*groups)
        print(f"\n  ANOVA on ensemble returns: F={f_stat:.4f}, p={p_val:.4f} -> {'SIGNIFICANT' if p_val < 0.05 else 'NOT SIGNIFICANT'}")
    except:
        print(f"\n  ANOVA: could not compute")

    # Pairwise vs no_nlp
    baseline = ens_df[ens_df['llm_config'] == 'no_nlp']['total_return'].values
    print(f"\n  Pairwise vs No-NLP baseline:")
    for cfg in LLM_CONFIGS[:-1]:
        test = ens_df[ens_df['llm_config'] == cfg]['total_return'].values
        if len(test) > 0 and len(baseline) > 0:
            diff = np.mean(test) - np.mean(baseline)
            try:
                t_stat, p_val = stats.ttest_ind(baseline, test)
                print(f"    {cfg:15s}: diff={diff*100:+.2f}pp, p={p_val:.4f} {'*' if p_val < 0.05 else ''}")
            except:
                print(f"    {cfg:15s}: diff={diff*100:+.2f}pp")

    # ── Also compare individual (non-ensemble) active-only results ──
    print(f"\n--- Individual Active-Only Results (for comparison) ---")
    active_train = train_df[train_df['is_active'] == True]
    print(f"{'Config':15s} {'N':>3s} {'MeanRet':>9s} {'StdRet':>8s} {'MeanSharpe':>11s}")
    print("-" * 50)
    for cfg in LLM_CONFIGS:
        sub = active_train[active_train['llm_config'] == cfg]
        if len(sub) > 0:
            print(f"  {cfg:13s} {len(sub):3d} {sub['total_return'].mean()*100:+8.2f}% {sub['total_return'].std()*100:7.2f}% {sub['sharpe_ratio'].mean():10.2f}")

    # Save report
    report = {
        "timestamp": datetime.now().isoformat(),
        "experiment": "LLM Data Impact v4 — Ensemble of Active Agents",
        "n_seeds": len(SEEDS),
        "seeds": SEEDS,
        "tickers": TICKERS,
        "llm_configs": LLM_CONFIGS,
        "total_timesteps": TOTAL_TIMESTEPS,
        "reward_type": REWARD_TYPE,
        "reward_scaling": REWARD_SCALING,
        "nlp_scale": NLP_SCALE,
        "active_threshold": ACTIVE_THRESHOLD,
        "obs_normalization": "selective: z-score cash/price/holdings + 25x NLP upscale",
        "ensemble_method": "mean action across active agents",
        "ensemble_results": {cfg: ens_df[ens_df['llm_config']==cfg][['ticker','total_return','sharpe_ratio','n_active']].to_dict('records') for cfg in LLM_CONFIGS},
    }
    with open(output_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    return ens_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, default=0, help="1=train only, 2=ensemble only, 0=both")
    parser.add_argument("--config", type=str, default=None, help="Single LLM config")
    parser.add_argument("--ticker", type=str, default=None, help="Single ticker")
    parser.add_argument("--seed", type=int, default=None, help="Single seed")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.config or args.ticker or args.seed:
        # Run single training
        cfg = args.config or LLM_CONFIGS[0]
        t = args.ticker or TICKERS[0]
        s = args.seed or SEEDS[0]
        train_single(cfg, t, s, OUTPUT_DIR)
    elif args.phase == 1:
        training_metrics = run_phase1(OUTPUT_DIR)
    elif args.phase == 2:
        ensemble_metrics = run_phase2(OUTPUT_DIR)
    else:
        training_metrics = run_phase1(OUTPUT_DIR)
        ensemble_metrics = run_phase2(OUTPUT_DIR)
        generate_report(training_metrics, ensemble_metrics, OUTPUT_DIR)
