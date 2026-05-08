#!/usr/bin/env python3
"""
LLM Impact Test — PPO with cash_penalty reward (PrimoGPT reference)
=====================================================================
Head-to-head comparison of 4 LLM signal sources + no-NLP baseline.
  - Gemini 3.1 Pro (closed-model upper bound)
  - Qwen3.5-27B base (open-model baseline)
  - Qwen3.5-27B + QLoRA (efficient fine-tuning)
  - Qwen3.5-27B + QA-LoRA (quantization-aware adaptation)
  - No-NLP (tech indicators only — ablation control)

Design decisions (aligned with project proposal & reference code):
  - FixedYahooDownloader (column-name-based, auto_adjust=True) — correct OHLC
  - NLP shift(1) to prevent look-ahead bias
  - 0.1% transaction costs (realistic)
  - PPO with cash_penalty reward, reward_scaling=100 (PrimoGPT reference)
  - The cash_penalty reward penalizes idle cash, forcing active trading
  - 5 seeds: [42, 123, 456, 789, 2024]
  - 5 test tickers: AAPL, AMZN, CRM, MSFT, NFLX
  - Train: 2022-04-01 to 2024-07-31, Test: 2024-08-01 to 2025-02-28
  - Filter not-learned agents (do_nothing > 95%) for active-only analysis
  - Ensemble of active agents per config-ticker
"""

import pandas as pd
import numpy as np
import os, sys, json, argparse, time, warnings
from datetime import datetime
from pathlib import Path

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
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.logger import configure

# ── Constants ──
SEEDS = [42, 123, 456, 789, 2024]
TICKERS = ["AAPL", "AMZN", "CRM", "MSFT", "NFLX"]
LLM_CONFIGS = ["gemini", "qwen_base", "qwen_qlora", "qwen_qalora", "no_nlp"]
TRAIN_START = "2022-04-01"
TRAIN_END = "2024-07-31"
TEST_START = "2024-08-01"
TEST_END = "2025-02-28"

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

# ── Reward Function ──
# cash_penalty = PrimoGPT reference reward (dollar_delta + dynamic cash penalty)
# Forces active trading, penalizes idle cash, adapts penalty to market trend.
# This is the reward used in env_primo_features_stocktrading.py (the results section).
# dollar_delta alone causes policy collapse to buy-and-hold in bull markets.
REWARD_TYPE = "cash_penalty"
REWARD_SCALING = 100
TRANSACTION_COST = 0.001  # 0.1%
TOTAL_TIMESTEPS = 400000

# PPO Hyperparameters (reference-aligned, matching PrimoGPT paper)
PPO_PARAMS = dict(
    n_steps=2048,
    ent_coef=0.01,
    learning_rate=0.00025,
    batch_size=128,
)

# Filter threshold: agents with >95% do-nothing days are "not learned"
DO_NOTHING_FILTER = 95.0


def load_nlp_data(llm_config, ticker):
    """Load NLP signal data for a given LLM config and ticker."""
    if llm_config == "no_nlp":
        return None

    if llm_config == "gemini":
        path = OUR_CODE / "ppo_data" / f"gemini_ppo_{ticker}_data.csv"
    elif llm_config == "qwen_base":
        path = OUR_CODE / "LLM_data" / f"qwen_base_ppo_{ticker}_data.csv"
    elif llm_config == "qwen_qlora":
        path = OUR_CODE / "LLM_data" / f"qwen_qlora_ppo_{ticker}_data.csv"
    elif llm_config == "qwen_qalora":
        path = OUR_CODE / "LLM_data" / f"qwen_qalora_ppo_{ticker}_data.csv"
    else:
        raise ValueError(f"Unknown LLM config: {llm_config}")

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

    # Ensure date is string format
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
        tech_indicator_list=["macd", "boll_ub", "boll_lb", "rsi_30",
                             "cci_30", "dx_30", "close_30_sma", "close_60_sma"],
        use_vix=True,
        use_turbulence=True,
        user_defined_feature=False,
    )
    processed = fe.preprocess_data(df)
    processed = processed.ffill().bfill().fillna(0)
    processed = processed.sort_values(["date", "tic"], ignore_index=True)
    return processed


def prepare_combined_data(price_df, nlp_df, llm_config, ticker):
    """Merge price + NLP data, apply shift(1), and split train/test."""
    if llm_config == "no_nlp":
        result = price_df.copy()
        for col in FUNDAMENTAL_INDICATORS:
            result[col] = 0.0
    else:
        result = price_df.merge(nlp_df, on="date", how="left")

        # Handle duplicate columns from merge
        for col in list(result.columns):
            if col.endswith("_x"):
                base = col[:-2]
                if base + "_y" in result.columns:
                    result[base] = result[col]
                    result = result.drop(columns=[col, base + "_y"])
            elif col.endswith("_y") and col[:-2] + "_x" not in result.columns:
                result = result.rename(columns={col: col[:-2]})

        # Drop leftover NLP columns
        drop_extra = [c for c in result.columns if c.endswith("_nlp")]
        result = result.drop(columns=drop_extra, errors="ignore")

        # Ensure all NLP columns exist
        for col in FUNDAMENTAL_INDICATORS:
            if col not in result.columns:
                result[col] = 0.0

        # shift(1) NLP features to prevent look-ahead bias
        for col in FUNDAMENTAL_INDICATORS:
            result[col] = result.groupby("tic")[col].shift(1)

        # Fill NaN from shift
        result = result.fillna(0)

    # Verify required columns
    required = ["date", "tic", "close"] + FUNDAMENTAL_INDICATORS
    for col in required:
        if col not in result.columns:
            raise ValueError(f"Missing required column: {col}")

    # Train/test split
    train = data_split(result, TRAIN_START, TRAIN_END)
    test = data_split(result, TEST_START, TEST_END)

    return train, test, result


def compute_metrics(account_values, risk_free_rate=0.03):
    """Compute comprehensive trading metrics from account value series."""
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
            "return_variance": 0, "final_value": final,
            "win_rate": 0, "profit_factor": 0,
            "max_consec_wins": 0, "max_consec_losses": 0,
        }

    # Sharpe ratio (annualized)
    mean_ret = np.mean(daily_returns)
    std_ret = np.std(daily_returns) + 1e-10
    daily_rf = risk_free_rate / 252
    sharpe = (mean_ret - daily_rf) / std_ret * np.sqrt(252)

    # Annualized return & volatility
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

    # Return variance (in basis points squared)
    return_variance = np.var(daily_returns) * 10000

    # Win rate (positive daily returns)
    win_rate = np.mean(daily_returns > 0) * 100

    # Profit factor
    gains = daily_returns[daily_returns > 0].sum()
    losses = abs(daily_returns[daily_returns < 0].sum())
    profit_factor = gains / losses if losses > 0 else float('inf')

    # Max consecutive wins/losses
    wins = (daily_returns > 0).astype(int)
    max_cw = max_cl = 0
    cw = cl = 0
    for w in wins:
        if w:
            cw += 1; cl = 0; max_cw = max(max_cw, cw)
        else:
            cl += 1; cw = 0; max_cl = max(max_cl, cl)

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "max_drawdown_duration": max_dd_duration,
        "volatility": volatility,
        "return_variance": return_variance,
        "final_value": final,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "max_consec_wins": max_cw,
        "max_consec_losses": max_cl,
    }


def compute_trade_metrics(actions_list):
    """Compute trade-level metrics from actions list."""
    position_changes = 0
    holding = 0  # 0 = no position, 1 = long

    for action in actions_list:
        if isinstance(action, np.ndarray):
            a = action[0] if len(action) > 0 else 0
        elif isinstance(action, (int, float)):
            a = action
        else:
            a = float(action)

        if a > 0:
            new_holding = 1
        elif a < 0:
            new_holding = 0
        else:
            new_holding = holding

        if new_holding != holding:
            position_changes += 1
        holding = new_holding

    total_days = len(actions_list)
    do_nothing_days = sum(1 for a in actions_list
                         if (a[0] if isinstance(a, np.ndarray) else a) == 0)
    do_nothing_pct = do_nothing_days / total_days * 100 if total_days > 0 else 0

    return {
        "num_trades": position_changes,
        "do_nothing_pct": do_nothing_pct,
    }


def train_and_eval(llm_config, ticker, seed, output_dir):
    """Train PPO with given LLM config, ticker, seed. Return metrics dict."""
    run_name = f"{llm_config}_{ticker}_seed{seed}"
    result_dir = output_dir / run_name
    result_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Training: {run_name}")
    print(f"  LLM: {llm_config} | Ticker: {ticker} | Seed: {seed}")
    print(f"  Reward: {REWARD_TYPE} x{REWARD_SCALING} | Timesteps: {TOTAL_TIMESTEPS}")
    print(f"{'='*60}")

    try:
        # Load data
        nlp_df = load_nlp_data(llm_config, ticker)
        price_df = load_price_data(ticker)
        train_df, test_df, full_df = prepare_combined_data(
            price_df, nlp_df, llm_config, ticker
        )

        # Set up environment
        from finrl.meta.env_primo_trading.env_primorl import StockTradingEnv

        stock_dim = 1
        state_space = (1 + 2 * stock_dim + 8 * stock_dim
                       + len(FUNDAMENTAL_INDICATORS) * stock_dim)

        env_kwargs = dict(
            hmax=1000,
            initial_amount=100000,
            buy_cost_pct=[TRANSACTION_COST],
            sell_cost_pct=[TRANSACTION_COST],
            num_stock_shares=[0],
            stock_dim=stock_dim,
            state_space=state_space,
            action_space=stock_dim,
            tech_indicator_list=["macd", "boll_ub", "boll_lb", "rsi_30",
                                  "cci_30", "dx_30", "close_30_sma", "close_60_sma"],
            fundamental_indicator_list=FUNDAMENTAL_INDICATORS,
            reward_type=REWARD_TYPE,
            reward_scaling=REWARD_SCALING,
            initial=True,
            print_verbosity=999999,
        )

        e_train_gym = StockTradingEnv(df=train_df, **env_kwargs)

        # Train PPO
        agent = DRLAgent(env=e_train_gym)

        model = agent.get_model(
            "ppo",
            model_kwargs=PPO_PARAMS,
            seed=seed,
        )

        # Suppress SB3 logger noise
        new_logger = configure(str(result_dir / "tb"),
                               ["stdout", "csv", "tensorboard"])
        model.set_logger(new_logger)

        # Train
        t0 = time.time()
        trained_model = agent.train_model(
            model=model,
            tb_log_name=f"ppo_{run_name}",
            total_timesteps=TOTAL_TIMESTEPS,
        )
        train_time = time.time() - t0
        print(f"  Training time: {train_time:.0f}s")

        # Evaluate on test data
        e_test_gym = StockTradingEnv(df=test_df, **env_kwargs)
        reset_result = e_test_gym.reset()
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result

        account_values = [e_test_gym.asset_memory[-1]]
        actions_list = []

        for _ in range(len(test_df.index.unique())):
            action, _ = trained_model.predict(obs, deterministic=True)
            action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
            action = np.clip(action, -1.0, 1.0)

            step_result = e_test_gym.step(action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, reward, done, info = step_result

            account_values.append(e_test_gym.asset_memory[-1])
            actions_list.append(action)
            if done:
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

        # Save account values and actions
        pd.DataFrame({"account_value": account_values}).to_csv(
            result_dir / "account_value.csv", index=False
        )
        pd.DataFrame({"actions": [str(a) for a in actions_list]}).to_csv(
            result_dir / "actions.csv", index=False
        )

        # Save metrics
        with open(result_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        print(f"  Return: {metrics['total_return']*100:+.2f}% | "
              f"Sharpe: {metrics['sharpe_ratio']:.2f} | "
              f"Trades: {metrics['num_trades']} | "
              f"DoNoth: {metrics['do_nothing_pct']:.1f}%")
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


def run_all(output_dir, configs=None, tickers=None, seeds=None):
    """Run all experiments."""
    configs = configs or LLM_CONFIGS
    tickers = tickers or TICKERS
    seeds = seeds or SEEDS

    all_metrics = []
    total = len(configs) * len(tickers) * len(seeds)
    done = 0

    for llm_config in configs:
        for ticker in tickers:
            for seed in seeds:
                done += 1
                print(f"\n[{done}/{total}] ", end="")
                m = train_and_eval(llm_config, ticker, seed, output_dir)
                all_metrics.append(m)

    # Save all results
    results_df = pd.DataFrame(all_metrics)
    results_df.to_csv(output_dir / "all_results.csv", index=False)

    # Generate summary
    summary = generate_summary(all_metrics, configs, tickers)

    # Compute ensemble
    ensemble = compute_ensemble(all_metrics, output_dir, configs, tickers)
    summary["ensemble_results"] = ensemble

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return all_metrics, summary


def generate_summary(all_metrics, configs, tickers):
    """Generate structured summary from all metrics."""
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
            "do_nothing_filter_pct": DO_NOTHING_FILTER,
        },
        "per_config_ticker": {},
        "per_config_overall": {},
    }

    for llm_config in configs:
        summary["per_config_ticker"][llm_config] = {}
        for ticker in tickers:
            runs = [m for m in all_metrics
                    if m.get("llm_config") == llm_config
                    and m.get("ticker") == ticker
                    and "error" not in m]
            if not runs:
                continue

            returns = [r["total_return"] for r in runs]
            sharpe = [r["sharpe_ratio"] for r in runs]
            trades = [r["num_trades"] for r in runs]
            do_nothing = [r["do_nothing_pct"] for r in runs]
            max_dd = [r["max_drawdown"] for r in runs]
            ret_var = [r.get("return_variance", 0) for r in runs]
            volatility = [r.get("volatility", 0) for r in runs]
            win_rate = [r.get("win_rate", 0) for r in runs]

            # Filter not-learned agents
            active_runs = [r for r in runs
                          if r.get("do_nothing_pct", 0) < DO_NOTHING_FILTER]
            active_returns = [r["total_return"] for r in active_runs]
            active_sharpe = [r["sharpe_ratio"] for r in active_runs]

            summary["per_config_ticker"][llm_config][ticker] = {
                "mean_return_pct": np.mean(returns) * 100,
                "std_return_pct": np.std(returns) * 100,
                "mean_sharpe": np.mean(sharpe),
                "mean_trades": np.mean(trades),
                "mean_do_nothing_pct": np.mean(do_nothing),
                "mean_max_drawdown_pct": np.mean(max_dd) * 100,
                "mean_return_variance": np.mean(ret_var),
                "mean_volatility": np.mean(volatility),
                "mean_win_rate": np.mean(win_rate),
                "n_active": len(active_runs),
                "n_total": len(runs),
                "active_mean_return_pct": np.mean(active_returns) * 100 if active_returns else 0,
                "active_std_return_pct": np.std(active_returns) * 100 if len(active_returns) > 1 else 0,
                "active_mean_sharpe": np.mean(active_sharpe) if active_sharpe else 0,
                "individual_returns_pct": [r * 100 for r in returns],
                "individual_sharpe": sharpe,
            }

    for llm_config in configs:
        runs = [m for m in all_metrics
                if m.get("llm_config") == llm_config and "error" not in m]
        if not runs:
            continue
        active_runs = [r for r in runs if r.get("do_nothing_pct", 0) < DO_NOTHING_FILTER]

        summary["per_config_overall"][llm_config] = {
            "mean_return_pct": np.mean([r["total_return"] for r in runs]) * 100,
            "active_mean_return_pct": np.mean([r["total_return"] for r in active_runs]) * 100 if active_runs else 0,
            "mean_sharpe": np.mean([r["sharpe_ratio"] for r in runs]),
            "active_mean_sharpe": np.mean([r["sharpe_ratio"] for r in active_runs]) if active_runs else 0,
            "mean_trades": np.mean([r["num_trades"] for r in runs]),
            "mean_do_nothing_pct": np.mean([r["do_nothing_pct"] for r in runs]),
            "mean_max_drawdown_pct": np.mean([r["max_drawdown"] for r in runs]) * 100,
            "active_rate_pct": len(active_runs) / len(runs) * 100 if runs else 0,
        }

    return summary


def compute_ensemble(all_metrics, output_dir, configs, tickers):
    """Compute ensemble: average account values of active agents per config-ticker."""
    ensemble_results = {}

    for llm_config in configs:
        ensemble_results[llm_config] = {}

        for ticker in tickers:
            runs = [m for m in all_metrics
                    if m.get("llm_config") == llm_config
                    and m.get("ticker") == ticker
                    and "error" not in m]

            active_runs = [r for r in runs
                          if r.get("do_nothing_pct", 0) < DO_NOTHING_FILTER]

            if not active_runs:
                ensemble_results[llm_config][ticker] = {
                    "ensemble_return_pct": 0,
                    "ensemble_sharpe": 0,
                    "ensemble_max_drawdown_pct": 0,
                    "ensemble_return_variance": 0,
                    "n_ensemble": 0,
                    "note": "No active agents",
                }
                continue

            # Load and average account values
            av_list = []
            for r in active_runs:
                seed = r["seed"]
                run_name = f"{llm_config}_{ticker}_seed{seed}"
                av_path = output_dir / run_name / "account_value.csv"
                if av_path.exists():
                    av = pd.read_csv(av_path)["account_value"].values
                    av_list.append(av)

            if av_list:
                min_len = min(len(a) for a in av_list)
                aligned = np.array([a[:min_len] for a in av_list])
                ensemble_av = aligned.mean(axis=0)

                em = compute_metrics(ensemble_av.tolist())
                ensemble_results[llm_config][ticker] = {
                    "ensemble_return_pct": em["total_return"] * 100,
                    "ensemble_sharpe": em["sharpe_ratio"],
                    "ensemble_max_drawdown_pct": em["max_drawdown"] * 100,
                    "ensemble_return_variance": em.get("return_variance", 0),
                    "ensemble_volatility": em.get("volatility", 0),
                    "ensemble_win_rate": em.get("win_rate", 0),
                    "n_ensemble": len(av_list),
                }

    return ensemble_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Impact Test")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--configs", nargs="+", default=None)
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else (
        OUR_CODE / "llm_impact_cashpenalty"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = args.configs or LLM_CONFIGS
    tickers = args.tickers or TICKERS
    seeds = args.seeds or SEEDS

    print(f"LLM Impact Test — PPO with {REWARD_TYPE} x{REWARD_SCALING}")
    print(f"Output: {output_dir}")
    print(f"Configs: {configs}")
    print(f"Tickers: {tickers}")
    print(f"Seeds: {seeds}")
    print(f"Reward: {REWARD_TYPE} | Scaling: {REWARD_SCALING}")
    print(f"Timesteps: {TOTAL_TIMESTEPS}")
    print(f"TX Cost: {TRANSACTION_COST*100}%")
    print()

    all_metrics, summary = run_all(output_dir, configs, tickers, seeds)

    # Print final comparison table
    print("\n" + "=" * 120)
    print("FINAL COMPARISON TABLE — All Seeds (Raw Mean)")
    print("=" * 120)
    hdr = (f"{'Config':<14} {'Ticker':<7} {'Return%':>9} {'Sharpe':>8} "
           f"{'Trades':>7} {'DoNoth%':>8} {'MaxDD%':>8} {'RetVar':>8} "
           f"{'Vol':>7} {'WinR%':>6} {'Active':>7}")
    print(hdr)
    print("-" * 120)
    for llm_config in configs:
        for ticker in tickers:
            key = llm_config
            if (key in summary["per_config_ticker"]
                    and ticker in summary["per_config_ticker"][key]):
                d = summary["per_config_ticker"][key][ticker]
                print(f"{llm_config:<14} {ticker:<7} "
                      f"{d['mean_return_pct']:>+8.2f}% "
                      f"{d['mean_sharpe']:>7.2f} "
                      f"{d['mean_trades']:>7.1f} "
                      f"{d['mean_do_nothing_pct']:>7.1f}% "
                      f"{d['mean_max_drawdown_pct']:>7.2f}% "
                      f"{d['mean_return_variance']:>7.2f} "
                      f"{d['mean_volatility']:>7.2f} "
                      f"{d['mean_win_rate']:>5.1f}% "
                      f"{d['n_active']}/{d['n_total']}")
        print()

    # Active-only table
    print("\n" + "=" * 120)
    print("ACTIVE-ONLY TABLE (filtered: do_nothing < 95%)")
    print("=" * 120)
    hdr = (f"{'Config':<14} {'Ticker':<7} {'Return%':>9} {'Sharpe':>8} "
           f"{'MaxDD%':>8} {'RetVar':>8} {'Vol':>7} {'WinR%':>6} "
           f"{'Active':>7}")
    print(hdr)
    print("-" * 120)
    for llm_config in configs:
        for ticker in tickers:
            key = llm_config
            if (key in summary["per_config_ticker"]
                    and ticker in summary["per_config_ticker"][key]):
                d = summary["per_config_ticker"][key][ticker]
                if d['n_active'] > 0:
                    print(f"{llm_config:<14} {ticker:<7} "
                          f"{d['active_mean_return_pct']:>+8.2f}% "
                          f"{d['active_mean_sharpe']:>7.2f} "
                          f"{d['mean_max_drawdown_pct']:>7.2f}% "
                          f"{d['mean_return_variance']:>7.2f} "
                          f"{d['mean_volatility']:>7.2f} "
                          f"{d['mean_win_rate']:>5.1f}% "
                          f"{d['n_active']}/{d['n_total']}")
        print()

    # Ensemble table
    print("\n" + "=" * 120)
    print("ENSEMBLE RESULTS (active agents, averaged account values)")
    print("=" * 120)
    hdr = (f"{'Config':<14} {'Ticker':<7} {'EnsRet%':>9} {'EnsSharpe':>10} "
           f"{'EnsMaxDD%':>10} {'EnsRetVar':>10} {'EnsVol':>8} "
           f"{'EnsWin%':>8} {'N_Ens':>6}")
    print(hdr)
    print("-" * 120)
    for llm_config in configs:
        if llm_config in summary["ensemble_results"]:
            for ticker in tickers:
                if ticker in summary["ensemble_results"][llm_config]:
                    d = summary["ensemble_results"][llm_config][ticker]
                    print(f"{llm_config:<14} {ticker:<7} "
                          f"{d.get('ensemble_return_pct',0):>+8.2f}% "
                          f"{d.get('ensemble_sharpe',0):>9.2f} "
                          f"{d.get('ensemble_max_drawdown_pct',0):>9.2f}% "
                          f"{d.get('ensemble_return_variance',0):>9.2f} "
                          f"{d.get('ensemble_volatility',0):>7.2f} "
                          f"{d.get('ensemble_win_rate',0):>7.1f}% "
                          f"{d.get('n_ensemble',0):>5}")
        print()

    print(f"\nResults saved to: {output_dir}")
