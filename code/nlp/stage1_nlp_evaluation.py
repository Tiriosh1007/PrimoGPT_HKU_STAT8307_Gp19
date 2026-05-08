#!/usr/bin/env python3
"""
Stage 1: NLP Signal Quality Evaluation
=======================================
Evaluates how well each LLM's extracted NLP signals predict next-day market direction.

Metrics:
1. Directional Accuracy: does the NLP signal direction match actual return direction?
2. Macro-F1: classify up/down/neutral vs actual movement
3. Per-feature Pearson correlation with next-day returns
4. Cross-LLM agreement: how much do LLMs agree on the same news?
5. Zero-fill rate analysis: how often is there no signal?
6. Information coefficient (IC): rank correlation between signal and returns
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
import json
import os
import warnings
warnings.filterwarnings("ignore")

# ── Config ──
TICKERS = ["AAPL", "AMZN", "CRM", "MSFT", "NFLX"]
NLP_COLS = ["News Relevance", "Sentiment", "Price Impact Potential",
            "Trend Direction", "Earnings Impact", "Investor Confidence",
            "Risk Profile Change"]
# Columns that have inherent directionality (positive = bullish signal)
DIRECTIONAL_COLS = ["Sentiment", "Price Impact Potential", "Trend Direction",
                    "Earnings Impact", "Investor Confidence"]

DATA_PATHS = {
    "gemini": "ppo_data/gemini_ppo_{t}_data.csv",
    "qwen_base": "LLM_data/qwen_base_ppo_{t}_data.csv",
    "qwen_qlora": "LLM_data/qwen_qlora_ppo_{t}_data.csv",
    "qwen_qalora": "LLM_data/qwen_qalora_ppo_{t}_data.csv",
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_data(config, ticker):
    fp = os.path.join(BASE_DIR, DATA_PATHS[config].format(t=ticker))
    df = pd.read_csv(fp, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    # Fill NaN with 0 (same as PPO training pipeline)
    nlp_cols_in_df = [c for c in NLP_COLS if c in df.columns]
    df[nlp_cols_in_df] = df[nlp_cols_in_df].fillna(0)
    # Add next-day return
    df["next_return"] = df["Returns"].shift(-1)
    # Direction labels: up/down/neutral
    df["actual_direction"] = df["next_return"].apply(
        lambda x: 1 if x > 0.005 else (-1 if x < -0.005 else 0) if pd.notna(x) else np.nan
    )
    return df


def directional_accuracy(df, col):
    """Does the NLP signal direction match the actual return direction?"""
    mask = (df[col] != 0) & df["actual_direction"].notna()
    if mask.sum() < 10:
        return {"n": int(mask.sum()), "accuracy": np.nan, "p_value": np.nan}

    signal_dir = np.sign(df.loc[mask, col])
    actual_dir = df.loc[mask, "actual_direction"]

    # For directional cols, positive signal should predict positive return
    # For risk_profile_change, negative = more risk = bearish, so flip
    if col == "Risk Profile Change":
        signal_dir = -signal_dir  # reduced risk (positive) = bullish

    agree = (signal_dir == actual_dir).mean()

    # Binomial test: H0 = 33.3% (random 3-way)
    n = mask.sum()
    k = (signal_dir == actual_dir).sum()
    # More appropriate: test against 50% for directional (non-zero signal should predict non-neutral direction)
    result_bt = stats.binomtest(k, n, 0.5, alternative='greater')
    p_val = result_bt.pvalue

    return {"n": int(n), "accuracy": round(agree, 4), "p_value": round(p_val, 4)}


def macro_f1(df, col):
    """F1 score for 3-class prediction (up/down/neutral)."""
    mask = (df[col] != 0) & df["actual_direction"].notna()
    if mask.sum() < 10:
        return {"n": int(mask.sum()), "f1": np.nan, "baseline_f1": np.nan}

    signal_dir = np.sign(df.loc[mask, col])
    if col == "Risk Profile Change":
        signal_dir = -signal_dir

    actual_dir = df.loc[mask, "actual_direction"].astype(int)

    # Only predict up/down from non-zero signals, map 0 to neutral
    pred = signal_dir.replace(0, 0)
    pred = pred.fillna(0).astype(int)

    f1 = f1_score(actual_dir, pred, average='macro', zero_division=0)
    # Baseline: always predict majority class
    majority = actual_dir.mode().iloc[0]
    baseline_pred = pd.Series(majority, index=actual_dir.index)
    baseline_f1 = f1_score(actual_dir, baseline_pred, average='macro', zero_division=0)

    return {"n": int(mask.sum()), "f1": round(f1, 4), "baseline_f1": round(baseline_f1, 4)}


def correlation_with_returns(df, col):
    """Pearson and Spearman correlation between NLP signal and next-day returns."""
    mask = df["next_return"].notna()
    if mask.sum() < 20:
        return {"n": int(mask.sum()), "pearson_r": np.nan, "pearson_p": np.nan,
                "spearman_r": np.nan, "spearman_p": np.nan, "ic": np.nan}

    x = df.loc[mask, col]
    y = df.loc[mask, "next_return"]

    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)
    # Information Coefficient (rank correlation, standard in quant finance)
    ic = sr

    return {"n": int(mask.sum()), "pearson_r": round(pr, 4), "pearson_p": round(pp, 6),
            "spearman_r": round(sr, 4), "spearman_p": round(sp, 6), "ic": round(ic, 4)}


def zero_fill_analysis(df, col):
    """Analyze how often the feature is zero-filled (no news / no signal)."""
    n = len(df)
    zero_count = (df[col] == 0).sum()
    nonzero_count = n - zero_count
    zero_pct = zero_count / n * 100

    # Max consecutive zeros
    is_zero = (df[col] == 0).values
    max_streak = 0
    current = 0
    for v in is_zero:
        if v:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0

    return {"total_days": n, "zero_days": int(zero_count), "nonzero_days": int(nonzero_count),
            "zero_pct": round(zero_pct, 1), "max_zero_streak": max_streak}


def cross_llm_agreement(ticker):
    """How much do different LLMs agree on the same news?"""
    dfs = {}
    for cfg in DATA_PATHS:
        df = load_data(cfg, ticker)
        dfs[cfg] = df.set_index("Date")[NLP_COLS]

    results = {}
    for col in NLP_COLS:
        # Build matrix: rows=dates, columns=LLMs
        mat = pd.DataFrame({cfg: dfs[cfg][col] for cfg in DATA_PATHS})
        mat = mat.dropna(how="any")

        # Pairwise Pearson correlation
        pair_corr = {}
        cfgs = list(DATA_PATHS.keys())
        for i in range(len(cfgs)):
            for j in range(i+1, len(cfgs)):
                a, b = cfgs[i], cfgs[j]
                mask = (mat[a] != 0) & (mat[b] != 0)
                if mask.sum() > 10:
                    r, p = stats.pearsonr(mat.loc[mask, a], mat.loc[mask, b])
                    pair_corr[f"{a}_vs_{b}"] = {"r": round(r, 4), "p": round(p, 6), "n": int(mask.sum())}
                else:
                    pair_corr[f"{a}_vs_{b}"] = {"r": np.nan, "p": np.nan, "n": int(mask.sum())}

        # Overall agreement: % of non-zero days where all LLMs agree on direction
        mask_all = (mat != 0).all(axis=1)
        if mask_all.sum() > 10:
            signs = np.sign(mat.loc[mask_all])
            agree_pct = (signs.nunique(axis=1) == 1).mean()
        else:
            agree_pct = np.nan

        results[col] = {"pairwise_correlation": pair_corr,
                       "direction_agreement_pct": round(agree_pct, 4) if not np.isnan(agree_pct) else np.nan,
                       "n_all_nonzero": int(mask_all.sum())}
    return results


def signal_vs_no_signal_returns(df):
    """Compare next-day returns when there IS a signal vs when there ISN'T."""
    results = {}
    for col in DIRECTIONAL_COLS:
        has_signal = df[col] != 0
        no_signal = df[col] == 0

        ret_with = df.loc[has_signal & df["next_return"].notna(), "next_return"]
        ret_without = df.loc[no_signal & df["next_return"].notna(), "next_return"]

        if len(ret_with) > 10 and len(ret_without) > 10:
            t_stat, p_val = stats.ttest_ind(ret_with, ret_without)
            results[col] = {
                "mean_ret_with_signal": round(ret_with.mean() * 100, 4),
                "mean_ret_without_signal": round(ret_without.mean() * 100, 4),
                "diff_bps": round((ret_with.mean() - ret_without.mean()) * 10000, 2),
                "t_stat": round(t_stat, 3),
                "p_value": round(p_val, 4),
                "n_with": len(ret_with),
                "n_without": len(ret_without)
            }
    return results


def run_full_evaluation():
    """Run the complete Stage 1 evaluation."""
    all_results = {}

    for cfg in DATA_PATHS:
        print(f"\n{'='*60}")
        print(f"  LLM Config: {cfg.upper()}")
        print(f"{'='*60}")
        cfg_results = {}

        for ticker in TICKERS:
            print(f"\n  --- {ticker} ---")
            df = load_data(cfg, ticker)
            ticker_results = {}

            # 1. Zero-fill analysis
            print(f"  Zero-fill rates:")
            zero_results = {}
            for col in NLP_COLS:
                zr = zero_fill_analysis(df, col)
                zero_results[col] = zr
                print(f"    {col}: {zr['zero_pct']:.1f}% zero (max streak: {zr['max_zero_streak']})")
            ticker_results["zero_fill"] = zero_results

            # 2. Directional accuracy
            print(f"\n  Directional accuracy (vs next-day return):")
            dir_results = {}
            for col in DIRECTIONAL_COLS:
                da = directional_accuracy(df, col)
                dir_results[col] = da
                sig = "***" if da.get("p_value", 1) < 0.001 else "**" if da.get("p_value", 1) < 0.01 else "*" if da.get("p_value", 1) < 0.05 else ""
                print(f"    {col}: {da['accuracy']:.1%} (n={da['n']}, p={da.get('p_value', 'N/A')}{sig})")
            ticker_results["directional_accuracy"] = dir_results

            # 3. Macro-F1
            print(f"\n  Macro-F1 (3-class: up/down/neutral):")
            f1_results = {}
            for col in DIRECTIONAL_COLS:
                f1r = macro_f1(df, col)
                f1_results[col] = f1r
                print(f"    {col}: F1={f1r['f1']:.3f} (baseline={f1r['baseline_f1']:.3f})")
            ticker_results["macro_f1"] = f1_results

            # 4. Correlation with returns
            print(f"\n  Correlation with next-day returns:")
            corr_results = {}
            for col in NLP_COLS:
                cr = correlation_with_returns(df, col)
                corr_results[col] = cr
                sig = "***" if cr.get("pearson_p", 1) < 0.001 else "**" if cr.get("pearson_p", 1) < 0.01 else "*" if cr.get("pearson_p", 1) < 0.05 else ""
                print(f"    {col}: r={cr['pearson_r']:.4f}, IC={cr['ic']:.4f}{sig}")
            ticker_results["correlation"] = corr_results

            # 5. Signal vs no-signal returns
            print(f"\n  Signal vs No-signal next-day returns:")
            sig_ret = signal_vs_no_signal_returns(df)
            for col, sr in sig_ret.items():
                print(f"    {col}: with={sr['mean_ret_with_signal']:.3f}%, without={sr['mean_ret_without_signal']:.3f}%, diff={sr['diff_bps']:.1f}bps (p={sr['p_value']:.3f})")
            ticker_results["signal_vs_nosignal"] = sig_ret

            cfg_results[ticker] = ticker_results

        all_results[cfg] = cfg_results

    # ── Cross-LLM Agreement ──
    print(f"\n{'='*60}")
    print(f"  CROSS-LLM AGREEMENT")
    print(f"{'='*60}")
    agreement_results = {}
    for ticker in TICKERS:
        print(f"\n  --- {ticker} ---")
        agr = cross_llm_agreement(ticker)
        agreement_results[ticker] = agr
        for col, data in agr.items():
            print(f"    {col}: direction_agreement={data.get('direction_agreement_pct', 'N/A')}, n_all_nonzero={data['n_all_nonzero']}")
            for pair, vals in data["pairwise_correlation"].items():
                if not np.isnan(vals.get("r", np.nan)):
                    print(f"      {pair}: r={vals['r']:.3f} (n={vals['n']})")

    # ── Aggregate Summary ──
    print(f"\n{'='*60}")
    print(f"  AGGREGATE SUMMARY (across all tickers)")
    print(f"{'='*60}")

    for cfg in DATA_PATHS:
        print(f"\n  {cfg.upper()}:")
        # Average directional accuracy across tickers
        accs = []
        f1s = []
        ics = []
        for ticker in TICKERS:
            for col in DIRECTIONAL_COLS:
                da = all_results[cfg][ticker]["directional_accuracy"][col]
                if not np.isnan(da["accuracy"]):
                    accs.append(da["accuracy"])
                f1r = all_results[cfg][ticker]["macro_f1"][col]
                if not np.isnan(f1r["f1"]):
                    f1s.append(f1r["f1"])
            for col in NLP_COLS:
                cr = all_results[cfg][ticker]["correlation"][col]
                if not np.isnan(cr["ic"]):
                    ics.append(cr["ic"])

        print(f"    Avg directional accuracy: {np.mean(accs):.1%} (random=33.3%)")
        print(f"    Avg Macro-F1: {np.mean(f1s):.3f}")
        print(f"    Avg Information Coefficient: {np.mean(ics):.4f}")
        print(f"    IC range: [{np.min(ics):.4f}, {np.max(ics):.4f}]")

    # Save full results
    out_path = os.path.join(BASE_DIR, "stage1_nlp_quality_results.json")
    # Convert numpy types for JSON
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(out_path, "w") as f:
        json.dump(convert({"per_config": all_results, "cross_llm_agreement": agreement_results}), f, indent=2)
    print(f"\n  Results saved to {out_path}")

    return all_results, agreement_results


if __name__ == "__main__":
    results, agreement = run_full_evaluation()
