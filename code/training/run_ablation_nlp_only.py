#!/usr/bin/env python3
"""
Ablation Study: NLP-Only Condition
====================================
Zeros out all 8 technical indicators so the PPO agent can only use
NLP signals + cash/price/holdings.

This directly tests: "Can NLP signals drive trading decisions
without tech indicator redundancy?"

Conditions compared:
  - Full (baseline): 8 tech + 7 NLP  [already exists in deep_30seed_results]
  - Tech-only: 8 tech + 0 NLP        [already exists as no_nlp in deep_30seed_results]
  - NLP-only: 0 tech + 7 NLP         [NEW - this script]

30 seeds x 5 tickers x 4 LLM configs = 600 runs
(no_nlp is meaningless for NLP-only condition, so we skip it)
"""

import sys, os, json, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths ──
OUR_CODE = Path("/Users/BryanMak/Documents/STAT8307 Natural Language Processing and Text Analytics/project/our_code")
PRIMO = Path("/Users/BryanMak/Documents/STAT8307 Natural Language Processing and Text Analytics/project/PrimoGPT-main")
sys.path.insert(0, str(PRIMO))

from finrl import config as finrl_config
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split

# ── Fixed YahooDownloader ──
class FixedYahooDownloader:
    def __init__(self, start_date, end_date, ticker_list):
        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list

    def fetch_data(self, proxy=None):
        import yfinance as yf
        data_df = pd.DataFrame()
        for tic in self.ticker_list:
            temp_df = yf.download(tic, start=self.start_date, end=self.end_date,
                                  multi_level_index=False, auto_adjust=True)
            temp_df["tic"] = tic
            if len(temp_df) > 0:
                data_df = pd.concat([data_df, temp_df], axis=0)
        data_df = data_df.reset_index()
        rename_map = {"Date":"date","Open":"open","High":"high","Low":"low",
                      "Close":"close","Volume":"volume"}
        data_df = data_df.rename(columns=rename_map)
        data_df["day"] = data_df["date"].dt.dayofweek
        data_df["date"] = data_df.date.apply(lambda x: x.strftime("%Y-%m-%d"))
        data_df = data_df.dropna().reset_index(drop=True)
        data_df = data_df.sort_values(["date","tic"]).reset_index(drop=True)
        return data_df

# ── Config ──
TRAIN_START = "2022-04-01"
TRAIN_END   = "2024-07-31"
TEST_START  = "2024-08-01"
TEST_END    = "2025-02-28"
TECH_INDICATORS = ["macd","boll_ub","boll_lb","rsi_30","cci_30","dx_30","close_30_sma","close_60_sma"]
FUNDAMENTAL_INDICATORS = [
    "news_relevance","sentiment","price_impact_potential",
    "trend_direction","earnings_impact","investor_confidence","risk_profile_change",
]
COLUMN_MAPPING = {
    "News Relevance":"news_relevance","Sentiment":"sentiment",
    "Price Impact Potential":"price_impact_potential","Trend Direction":"trend_direction",
    "Earnings Impact":"earnings_impact","Investor Confidence":"investor_confidence",
    "Risk Profile Change":"risk_profile_change",
}

TRANSACTION_COST = 0.001
TOTAL_TIMESTEPS = 200000
REWARD_TYPE = "dollar_delta"
REWARD_SCALING = 1
NLP_SCALE = 25.0
ACTIVE_THRESHOLD = 0.001

HP_DEEP = {
    "n_steps": 2048, "ent_coef": 0.01, "learning_rate": 1e-4,
    "batch_size": 256, "net_arch": [128, 128, 64],
}

# NLP-only: only LLM configs that HAVE NLP data (skip no_nlp)
LLM_CONFIGS = ["gemini","qwen_base","qwen_qlora","qwen_qalora"]
TICKERS = ["AAPL","AMZN","CRM","MSFT","NFLX"]
SEEDS_30 = [42,123,456,789,2024,314,271,1618,999,7,
            100,200,300,400,500,600,700,800,900,1000,
            1111,2222,3333,4444,5555,6666,7777,8888,9999,12345]

OUTPUT_DIR = OUR_CODE / "ablation_nlp_only_results"

# ── Data Loading (same as baseline) ──
def load_nlp_data(llm_config, ticker):
    if llm_config == "gemini":
        path = OUR_CODE / "ppo_data" / f"gemini_ppo_{ticker}_data.csv"
    else:
        path = OUR_CODE / "LLM_data" / f"{llm_config}_ppo_{ticker}_data.csv"
    if not path.exists():
        raise FileNotFoundError(f"NLP data not found: {path}")
    df = pd.read_csv(path)
    if "Date" in df.columns:
        df = df.rename(columns={"Date":"date"})
    for old_name, new_name in COLUMN_MAPPING.items():
        if old_name in df.columns:
            df = df.rename(columns={old_name: new_name})
    drop_cols = ["Adj Close Price","Returns","Bin Label","Prompt","ticker",
                 "Parse Success","Parse Error","Raw Response","close_nlp"]
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
    processed = processed.replace([np.inf, -np.inf], np.nan)
    processed = processed.ffill().bfill().fillna(0)
    processed = processed.sort_values(["date","tic"], ignore_index=True)
    return processed

def prepare_nlp_only_data(price_df, nlp_df, llm_config, ticker):
    """
    Like prepare_combined_data but ZEROS OUT all tech indicators.
    Agent sees: cash + price/holdings + 0_tech + NLP
    """
    # First do the normal merge
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

    # ABLATION: Zero out all tech indicators
    for col in TECH_INDICATORS:
        if col in result.columns:
            result[col] = 0.0

    train = data_split(result, TRAIN_START, TRAIN_END)
    test = data_split(result, TEST_START, TEST_END)
    return train, test, result

# ── Normalization Wrapper ──
import gymnasium as gym

class NormalizedObsWrapper(gym.Wrapper):
    def __init__(self, env, obs_mean=None, obs_std=None, nlp_scale=25.0, eps=1e-8):
        super().__init__(env)
        self.nlp_scale = nlp_scale
        self.eps = eps
        stock_dim = env.stock_dim
        n_tech = len(env.tech_indicator_list) * stock_dim
        n_nlp = len(env.fundamental_indicator_list) * stock_dim
        self.nlp_start = 1 + 2 * stock_dim + n_tech
        self.nlp_end = self.nlp_start + n_nlp
        self.zscore_indices = list(range(1 + 2 * stock_dim))
        self.obs_mean = obs_mean
        self.obs_std = obs_std

    def compute_stats_from_data(self, train_df):
        from finrl.meta.env_primo_trading.env_primorl import StockTradingEnv
        stock_dim = 1
        state_space = 1 + 2*stock_dim + len(TECH_INDICATORS)*stock_dim + len(FUNDAMENTAL_INDICATORS)*stock_dim
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
            action = np.zeros(stock_dim)
            obs, _, terminated, truncated, _ = temp_env.step(action)
            done = terminated or truncated
            if not done:
                all_obs.append(np.array(obs, dtype=np.float64))
        all_obs = np.array(all_obs)
        self.obs_mean = np.mean(all_obs, axis=0).astype(np.float32)
        self.obs_std = np.std(all_obs, axis=0).astype(np.float32)
        self.obs_std = np.maximum(self.obs_std, self.eps)
        return self.obs_mean, self.obs_std

    def _transform(self, obs):
        obs = np.array(obs, dtype=np.float32)
        if self.obs_mean is not None and self.obs_std is not None:
            for idx in self.zscore_indices:
                obs[idx] = (obs[idx] - self.obs_mean[idx]) / self.obs_std[idx]
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

# ── Env helpers ──
def get_env_kwargs():
    stock_dim = 1
    state_space = 1 + 2*stock_dim + len(TECH_INDICATORS)*stock_dim + len(FUNDAMENTAL_INDICATORS)*stock_dim
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

# ── Metrics ──
def compute_metrics(account_values, risk_free_rate=0.03):
    values = np.array(account_values)
    n = len(values)
    initial = values[0]
    final = values[-1]
    total_return = (final - initial) / initial
    daily_returns = np.diff(values) / values[:-1]
    daily_returns = daily_returns[~np.isnan(daily_returns)]
    if len(daily_returns) == 0:
        return {"total_return":0,"annualized_return":0,"sharpe_ratio":0,
                "max_drawdown":0,"max_drawdown_duration":0,"volatility":0,"final_value":final}
    mean_ret = np.mean(daily_returns)
    std_ret = np.std(daily_returns) + 1e-10
    daily_rf = risk_free_rate / 252
    sharpe = (mean_ret - daily_rf) / std_ret * np.sqrt(252)
    annualized_return = (1 + total_return) ** (252 / max(n-1,1)) - 1
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
        "total_return": total_return, "annualized_return": annualized_return,
        "sharpe_ratio": sharpe, "max_drawdown": max_dd,
        "max_drawdown_duration": max_dd_duration, "volatility": volatility,
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

# ── Training ──
def train_single(llm_config, ticker, seed, output_dir):
    from finrl.meta.env_primo_trading.env_primorl import StockTradingEnv
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    run_name = f"nlponly_{llm_config}_{ticker}_seed{seed}"
    result_dir = output_dir / "models" / run_name
    result_dir.mkdir(parents=True, exist_ok=True)

    if (result_dir / "metrics.json").exists():
        with open(result_dir / "metrics.json") as f:
            metrics = json.load(f)
        is_active = abs(metrics.get("total_return", 0)) > ACTIVE_THRESHOLD
        return is_active, metrics

    try:
        nlp_df = load_nlp_data(llm_config, ticker)
        price_df = load_price_data(ticker)
        train_df, test_df, full_df = prepare_nlp_only_data(price_df, nlp_df, llm_config, ticker)
        env_kwargs = get_env_kwargs()

        e_temp = StockTradingEnv(df=train_df, **env_kwargs)
        norm_temp = NormalizedObsWrapper(e_temp)
        obs_mean, obs_std = norm_temp.compute_stats_from_data(train_df)
        np.savez(result_dir / "norm_stats.npz", mean=obs_mean, std=obs_std)

        def make_train_env():
            raw = StockTradingEnv(df=train_df, **env_kwargs)
            return NormalizedObsWrapper(raw, obs_mean=obs_mean, obs_std=obs_std)

        train_env = DummyVecEnv([make_train_env])

        model = PPO(
            "MlpPolicy", train_env,
            n_steps=HP_DEEP["n_steps"], ent_coef=HP_DEEP["ent_coef"],
            learning_rate=HP_DEEP["learning_rate"], batch_size=HP_DEEP["batch_size"],
            policy_kwargs=dict(net_arch=HP_DEEP["net_arch"]),
            seed=seed, verbose=0, device="auto",
        )
        t0 = time.time()
        model.learn(total_timesteps=TOTAL_TIMESTEPS)
        train_time = time.time() - t0

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
        metrics["ablation"] = "nlp_only"

        model.save(str(result_dir / "model"))
        with open(result_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        is_active = abs(metrics.get("total_return", 0)) > ACTIVE_THRESHOLD
        ret_str = f"{metrics['total_return']*100:+.2f}%"
        print(f"  {run_name}: {ret_str} | Sharpe {metrics['sharpe_ratio']:.2f} | {'ACTIVE' if is_active else 'DO-NOTHING'}")
        return is_active, metrics

    except Exception as e:
        print(f"  {run_name}: ERROR - {e}")
        import traceback
        traceback.print_exc()
        return False, {"llm_config":llm_config,"ticker":ticker,"seed":seed,"error":str(e),
                       "total_return":0,"sharpe_ratio":0,"num_trades":0,"ablation":"nlp_only"}

# ── Ensemble Evaluation ──
def evaluate_ensemble(llm_config, ticker, output_dir):
    from finrl.meta.env_primo_trading.env_primorl import StockTradingEnv
    from stable_baselines3 import PPO

    print(f"\n  Ensemble: {llm_config} / {ticker}")
    active_models = []
    obs_mean = None
    obs_std = None

    for seed in SEEDS_30:
        run_name = f"nlponly_{llm_config}_{ticker}_seed{seed}"
        model_dir = output_dir / "models" / run_name
        metrics_path = model_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        with open(metrics_path) as f:
            metrics = json.load(f)
        if abs(metrics.get("total_return", 0)) <= ACTIVE_THRESHOLD:
            print(f"    Skip seed {seed}: do-nothing ({metrics.get('total_return',0)*100:+.2f}%)")
            continue
        if metrics.get("sharpe_ratio", 0) < 0 and metrics.get("num_trades", 0) > 50:
            print(f"    Skip seed {seed}: oscillating (Sharpe={metrics.get('sharpe_ratio',0):.1f}, trades={metrics.get('num_trades',0)})")
            continue
        model_path = model_dir / "model"
        if not Path(str(model_path) + ".zip").exists():
            continue
        model = PPO.load(str(model_path))
        if obs_mean is None:
            stats = np.load(model_dir / "norm_stats.npz")
            obs_mean = stats["mean"]
            obs_std = stats["std"]
        active_models.append((seed, model))

    n_active = len(active_models)
    print(f"    Active models: {n_active}/{len(SEEDS_30)}")

    if n_active == 0:
        return {"llm_config":llm_config,"ticker":ticker,"n_active":0,
                "total_return":0,"sharpe_ratio":0,"num_trades":0,"error":"No active models"}

    nlp_df = load_nlp_data(llm_config, ticker)
    price_df = load_price_data(ticker)
    train_df, test_df, full_df = prepare_nlp_only_data(price_df, nlp_df, llm_config, ticker)
    env_kwargs = get_env_kwargs()

    e_test_raw = StockTradingEnv(df=test_df, **env_kwargs)
    e_test = NormalizedObsWrapper(e_test_raw, obs_mean=obs_mean, obs_std=obs_std)

    obs, _ = e_test.reset()
    account_values = [e_test.unwrapped.asset_memory[-1]]
    ensemble_actions = []

    n_steps = len(test_df.index.unique())
    for step in range(n_steps):
        actions = []
        for seed, model in active_models:
            action, _ = model.predict(obs, deterministic=True)
            action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
            action = np.clip(action, -1.0, 1.0)
            actions.append(action)
        ensemble_action = np.mean(actions, axis=0)
        ensemble_actions.append(ensemble_action)
        obs, reward, terminated, truncated, info = e_test.step(ensemble_action)
        account_values.append(e_test.unwrapped.asset_memory[-1])
        if terminated or truncated:
            break

    metrics = compute_metrics(account_values)
    trade_metrics = compute_trade_metrics(ensemble_actions)
    metrics.update(trade_metrics)
    metrics["llm_config"] = llm_config
    metrics["ticker"] = ticker
    metrics["n_active"] = n_active
    metrics["n_total_seeds"] = len(SEEDS_30)
    metrics["ablation"] = "nlp_only"

    ens_dir = output_dir / "ensemble" / f"{llm_config}_{ticker}"
    ens_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"account_value": account_values}).to_csv(ens_dir / "account_value.csv", index=False)
    with open(ens_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    ret_str = f"{metrics['total_return']*100:+.2f}%"
    print(f"    Result: {ret_str} | Sharpe {metrics['sharpe_ratio']:.2f} | Trades {metrics['num_trades']}")
    return metrics

# ── Main ──
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train","ensemble","all"], default="all")
    parser.add_argument("--llm", type=str, default=None, help="Single LLM config to run")
    parser.add_argument("--ticker", type=str, default=None, help="Single ticker to run")
    parser.add_argument("--seed", type=int, default=None, help="Single seed to run")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Single-run mode (for parallel dispatch)
    if args.llm and args.ticker and args.seed is not None:
        is_active, metrics = train_single(args.llm, args.ticker, args.seed, OUTPUT_DIR)
        ret_str = f"{metrics.get('total_return',0)*100:+.2f}%"
        print(f"DONE: nlponly_{args.llm}_{args.ticker}_seed{args.seed}: {ret_str} | {'ACTIVE' if is_active else 'DO-NOTHING'}")
        sys.exit(0)

    # Filter configs if specified
    llm_configs = [args.llm] if args.llm else LLM_CONFIGS
    tickers = [args.ticker] if args.ticker else TICKERS
    seeds = [args.seed] if args.seed is not None else SEEDS_30

    if args.mode in ["train", "all"]:
        print("=" * 80)
        print("ABLATION: NLP-ONLY — 30 Seed PPO Training")
        print(f"  {len(SEEDS_30)} seeds x {len(TICKERS)} tickers x {len(LLM_CONFIGS)} LLM configs = {len(SEEDS_30)*len(TICKERS)*len(LLM_CONFIGS)} runs")
        print("=" * 80)

        from concurrent.futures import ProcessPoolExecutor, as_completed

        # Build task list
        tasks = []
        for llm_config in LLM_CONFIGS:
            for ticker in TICKERS:
                for seed in SEEDS_30:
                    tasks.append((llm_config, ticker, seed))

        # Check which already done
        remaining = []
        for llm_config, ticker, seed in tasks:
            run_name = f"nlponly_{llm_config}_{ticker}_seed{seed}"
            if not (OUTPUT_DIR / "models" / run_name / "metrics.json").exists():
                remaining.append((llm_config, ticker, seed))

        print(f"  Remaining: {len(remaining)}/{len(tasks)}")
        n_active = 0
        n_total = 0

        # Run sequentially with parallelism within each ticker group
        # (can't easily parallelize across processes due to gym env issues)
        for llm_config in LLM_CONFIGS:
            print(f"\n--- {llm_config.upper()} ---")
            for ticker in TICKERS:
                print(f"\n  {ticker}:")
                for seed in SEEDS_30:
                    is_active, metrics = train_single(llm_config, ticker, seed, OUTPUT_DIR)
                    n_total += 1
                    if is_active:
                        n_active += 1

        print(f"\n{'='*80}")
        print(f"TRAINING COMPLETE: {n_active}/{n_total} active agents ({n_active/n_total*100:.1f}%)")
        print(f"{'='*80}")

    if args.mode in ["ensemble", "all"]:
        print("\n" + "=" * 80)
        print("ABLATION: NLP-ONLY — Ensemble Evaluation")
        print("=" * 80)

        ensemble_results = []
        for llm_config in LLM_CONFIGS:
            for ticker in TICKERS:
                metrics = evaluate_ensemble(llm_config, ticker, OUTPUT_DIR)
                ensemble_results.append(metrics)

        # Save summary
        summary_path = OUTPUT_DIR / "ensemble_summary.json"
        with open(summary_path, "w") as f:
            json.dump(ensemble_results, f, indent=2, default=str)

        # Print comparison table
        print("\n" + "=" * 80)
        print("ABLATION COMPARISON: Full vs Tech-Only vs NLP-Only")
        print("=" * 80)

        # Load baseline (Full) and Tech-only (no_nlp) results
        baseline_dir = OUR_CODE / "deep_30seed_results"
        baseline_results = {}
        no_nlp_results = {}

        if baseline_dir.exists():
            ens_dir = baseline_dir / "ensemble"
            if ens_dir.exists():
                for d in os.listdir(ens_dir):
                    mp = ens_dir / d / "metrics.json"
                    if mp.exists():
                        with open(mp) as f:
                            m = json.load(f)
                        key = f"{m['llm_config']}_{m['ticker']}"
                        if m['llm_config'] == 'no_nlp':
                            no_nlp_results[key] = m
                        else:
                            baseline_results[key] = m

        # Print header
        print(f"\n{'Ticker':<8} {'LLM Config':<14} {'Full':>10} {'Tech-Only':>10} {'NLP-Only':>10} {'Full Sharpe':>11} {'Tech Sharpe':>11} {'NLP Sharpe':>10}")
        print("-" * 90)

        for llm_config in LLM_CONFIGS:
            for ticker in TICKERS:
                key = f"{llm_config}_{ticker}"
                bl = baseline_results.get(key, {})
                tn = no_nlp_results.get(f"no_nlp_{ticker}", {})
                nl = next((r for r in ensemble_results if r.get("llm_config")==llm_config and r.get("ticker")==ticker), {})

                bl_ret = f"{bl.get('total_return',0)*100:+.1f}%" if bl else "N/A"
                tn_ret = f"{tn.get('total_return',0)*100:+.1f}%" if tn else "N/A"
                nl_ret = f"{nl.get('total_return',0)*100:+.1f}%" if nl else "N/A"
                bl_sh = f"{bl.get('sharpe_ratio',0):.2f}" if bl else "N/A"
                tn_sh = f"{tn.get('sharpe_ratio',0):.2f}" if tn else "N/A"
                nl_sh = f"{nl.get('sharpe_ratio',0):.2f}" if nl else "N/A"

                print(f"{ticker:<8} {llm_config:<14} {bl_ret:>10} {tn_ret:>10} {nl_ret:>10} {bl_sh:>11} {tn_sh:>11} {nl_sh:>10}")
