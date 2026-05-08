#!/usr/bin/env python3
"""
Ablation Study: Complete Analysis
===================================
Compares three feature conditions:
  1. Full (8 tech + 7 NLP) - baseline from deep_30seed_results
  2. Tech-only (8 tech + 0 NLP) - no_nlp from deep_30seed_results
  3. NLP-only (0 tech + 7 NLP) - new ablation_nlp_only_results

Run AFTER training completes.
"""
import json, os, sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from scipy import stats

OUR_CODE = Path("/Users/BryanMak/Documents/STAT8307 Natural Language Processing and Text Analytics/project/our_code")
BASE_FULL = OUR_CODE / "deep_30seed_results" / "models"
BASE_NLP = OUR_CODE / "ablation_nlp_only_results" / "models"

LLM_CONFIGS = ["gemini", "qwen_base", "qwen_qlora", "qwen_qalora"]
TICKERS = ["AAPL", "AMZN", "CRM", "MSFT", "NFLX"]
SEEDS = [42,123,456,789,2024,314,271,1618,999,7,
         100,200,300,400,500,600,700,800,900,1000,
         1111,2222,3333,4444,5555,6666,7777,8888,9999,12345]

def load_results(base_dir, prefix, llm_field="llm_config"):
    """Load all metrics from a results directory."""
    results = []
    if not base_dir.exists():
        return results
    for d in sorted(os.listdir(base_dir)):
        if not d.startswith(prefix):
            continue
        mp = base_dir / d / "metrics.json"
        if not mp.exists():
            continue
        with open(mp) as f:
            m = json.load(f)
        results.append(m)
    return results

def is_active(m):
    """Filter: active (not do-nothing, not oscillating)."""
    return (abs(m.get('total_return', 0)) > 0.001 and 
            not (m.get('sharpe_ratio', 0) < 0 and m.get('num_trades', 0) > 50))

def main():
    print("=" * 90)
    print("ABLATION STUDY: Full vs Tech-Only vs NLP-Only")
    print("=" * 90)
    
    # Load data
    full_all = load_results(BASE_FULL, "deep_")
    nlp_all = load_results(BASE_NLP, "nlponly_")
    
    # Separate no_nlp (tech-only)
    tech_all = [m for m in full_all if m.get('llm_config') == 'no_nlp']
    full_all = [m for m in full_all if m.get('llm_config') != 'no_nlp']
    
    full_active = [m for m in full_all if is_active(m)]
    tech_active = [m for m in tech_all if is_active(m)]
    nlp_active = [m for m in nlp_all if is_active(m)]
    
    print(f"\nData loaded:")
    print(f"  Full: {len(full_all)} total, {len(full_active)} active ({len(full_active)/max(len(full_all),1)*100:.1f}%)")
    print(f"  Tech-only: {len(tech_all)} total, {len(tech_active)} active ({len(tech_active)/max(len(tech_all),1)*100:.1f}%)")
    print(f"  NLP-only: {len(nlp_all)} total, {len(nlp_active)} active ({len(nlp_active)/max(len(nlp_all),1)*100:.1f}%)")
    
    # ── Per-config comparison ──
    print(f"\n{'='*90}")
    print("PER-CONFIG COMPARISON (active agents, avg return)")
    print(f"{'='*90}")
    print(f"{'Config':<14} {'Ticker':<7} {'Full':>10} {'Tech-Only':>10} {'NLP-Only':>10} {'F vs N':>8} {'N vs T':>8} {'n_F':>4} {'n_T':>4} {'n_N':>4}")
    print("-" * 90)
    
    for config in LLM_CONFIGS:
        for ticker in TICKERS:
            f_rets = [m['total_return']*100 for m in full_active if m['llm_config']==config and m['ticker']==ticker]
            t_rets = [m['total_return']*100 for m in tech_active if m['ticker']==ticker]
            n_rets = [m['total_return']*100 for m in nlp_active if m.get('llm_config')==config and m['ticker']==ticker]
            
            f_avg = np.mean(f_rets) if f_rets else np.nan
            t_avg = np.mean(t_rets) if t_rets else np.nan
            n_avg = np.mean(n_rets) if n_rets else np.nan
            
            gap_fn = f_avg - n_avg if not np.isnan(n_avg) else np.nan
            gap_nt = n_avg - t_avg if not np.isnan(n_avg) else np.nan
            
            print(f"{config:<14} {ticker:<7} {f_avg:>+9.1f}% {t_avg:>+9.1f}% {n_avg:>+9.1f}% {gap_nt:>+7.1f}pp {gap_nt:>+7.1f}pp {len(f_rets):>4} {len(t_rets):>4} {len(n_rets):>4}")
    
    # ── Aggregate across tickers ──
    print(f"\n{'='*90}")
    print("AGGREGATE (across all tickers)")
    print(f"{'='*90}")
    
    for config in LLM_CONFIGS:
        f_rets = [m['total_return']*100 for m in full_active if m['llm_config']==config]
        t_rets = [m['total_return']*100 for m in tech_active]
        n_rets = [m['total_return']*100 for m in nlp_active if m.get('llm_config')==config]
        
        print(f"\n  {config.upper()}:")
        print(f"    Full:      {np.mean(f_rets):+.1f}% (n={len(f_rets)})")
        print(f"    Tech-only: {np.mean(t_rets):+.1f}% (n={len(t_rets)})")
        print(f"    NLP-only:  {np.mean(n_rets):+.1f}% (n={len(n_rets)})")
        if n_rets and f_rets:
            # Paired comparison (same seeds)
            f_by_key = {(m['llm_config'], m['ticker'], m['seed']): m['total_return']*100 
                       for m in full_active if m['llm_config']==config}
            n_by_key = {(m.get('llm_config'), m['ticker'], m['seed']): m['total_return']*100 
                       for m in nlp_active if m.get('llm_config')==config}
            common = set(f_by_key.keys()) & set(n_by_key.keys())
            if common:
                f_paired = [f_by_key[k] for k in common]
                n_paired = [n_by_key[k] for k in common]
                diffs = [n - f for n, f in zip(n_paired, f_paired)]
                t_stat, p_val = stats.ttest_rel(n_paired, f_paired)
                d = np.mean(diffs) / (np.std(diffs, ddof=1) + 1e-10)
                print(f"    Paired NLP-only vs Full: diff={np.mean(diffs):+.1f}pp, t={t_stat:.2f}, p={p_val:.4f}, d={d:.3f}")
    
    # ── ANOVA across conditions ──
    print(f"\n{'='*90}")
    print("ANOVA: Does feature condition affect trading performance?")
    print(f"{'='*90}")
    
    for config in LLM_CONFIGS:
        f_rets = [m['total_return']*100 for m in full_active if m['llm_config']==config]
        t_rets = [m['total_return']*100 for m in tech_active]
        n_rets = [m['total_return']*100 for m in nlp_active if m.get('llm_config')==config]
        
        if n_rets and f_rets and t_rets:
            f_stat, p_val = stats.f_oneway(f_rets, t_rets, n_rets)
            print(f"  {config}: F={f_stat:.3f}, p={p_val:.4f} {'***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'}")
    
    # ── Active rate comparison ──
    print(f"\n{'='*90}")
    print("ACTIVE RATE COMPARISON")
    print(f"{'='*90}")
    
    for config in LLM_CONFIGS:
        f_all_c = [m for m in full_all if m['llm_config']==config]
        n_all_c = [m for m in nlp_all if m.get('llm_config')==config]
        f_act_c = [m for m in f_all_c if is_active(m)]
        n_act_c = [m for m in n_all_c if is_active(m)]
        
        print(f"  {config}: Full={len(f_act_c)}/{len(f_all_c)} ({len(f_act_c)/max(len(f_all_c),1)*100:.1f}%), "
              f"NLP-only={len(n_act_c)}/{len(n_all_c)} ({len(n_act_c)/max(len(n_all_c),1)*100:.1f}%)")
    
    print(f"  Tech-only: {len(tech_active)}/{len(tech_all)} ({len(tech_active)/max(len(tech_all),1)*100:.1f}%)")
    
    # ── Key finding summary ──
    print(f"\n{'='*90}")
    print("KEY FINDING")
    print(f"{'='*90}")
    
    all_f = [m['total_return']*100 for m in full_active]
    all_t = [m['total_return']*100 for m in tech_active]
    all_n = [m['total_return']*100 for m in nlp_active]
    
    print(f"  Full (8 tech + 7 NLP): {np.mean(all_f):+.1f}% (n={len(all_f)})")
    print(f"  Tech-only (8 tech + 0 NLP): {np.mean(all_t):+.1f}% (n={len(all_t)})")
    print(f"  NLP-only (0 tech + 7 NLP): {np.mean(all_n):+.1f}% (n={len(all_n)})")
    print()
    print(f"  NLP-only vs Tech-only gap: {np.mean(all_n) - np.mean(all_t):+.1f}pp")
    print(f"  NLP-only vs Full gap: {np.mean(all_n) - np.mean(all_f):+.1f}pp")
    print()
    if np.mean(all_n) > np.mean(all_t):
        print("  ==> NLP signals alone outperform tech indicators alone!")
        print("  ==> The 'no effect' finding is due to REDUNDANCY, not irrelevance.")
    else:
        print("  ==> Tech indicators outperform NLP signals alone.")
        print("  ==> NLP provides marginal value beyond tech indicators.")

if __name__ == "__main__":
    main()
