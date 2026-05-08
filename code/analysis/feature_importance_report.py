#!/usr/bin/env python3
"""
Final Summary: Feature Importance Analysis for PPO Trading Models
=================================================================
This script generates a comprehensive summary report of the analysis.
"""

print("""
========================================================================
  FEATURE IMPORTANCE ANALYSIS - FINAL REPORT
  Determining Whether NLP Features Are Used by the PPO Policy
========================================================================

1. METHODOLOGY
==============
  Three methods were used to assess feature importance:

  a) Weight Analysis: Absolute magnitude of first-layer weights in the
     policy network (mlp_extractor.policy_net.0.weight, shape 64x18).
     This measures how strongly each input feature connects to the
     first hidden layer.

  b) Permutation Importance (GOLD STANDARD): For each feature, shuffle
     that feature across test observations and measure how much the
     policy's predicted actions change. Larger change = more important.
     20 repeats per feature for stability.

  c) Scale Analysis: Examine the raw value ranges of features as they
     enter the environment, to identify normalization issues.

  Models analyzed (retrained with identical configs to impact tests):
  - gemini_AAPL_seed2024
  - gemini_MSFT_seed2024
  - no_nlp_AAPL_seed2024

2. STATE SPACE STRUCTURE (18 dimensions)
=========================================
  Index  Feature                  Group
  -----  -------                  -----
  0      cash                     cash_price_holdings
  1      close_price              cash_price_holdings
  2      holdings                 cash_price_holdings
  3      macd                     tech_indicators
  4      boll_ub                  tech_indicators
  5      boll_lb                  tech_indicators
  6      rsi_30                   tech_indicators
  7      cci_30                   tech_indicators
  8      dx_30                    tech_indicators
  9      close_30_sma             tech_indicators
  10     close_60_sma             tech_indicators
  11     news_relevance           nlp_features
  12     sentiment                nlp_features
  13     price_impact_potential   nlp_features
  14     trend_direction          nlp_features
  15     earnings_impact          nlp_features
  16     investor_confidence      nlp_features
  17     risk_profile_change      nlp_features

3. RESULTS: Permutation Importance (Action Change When Feature Shuffled)
========================================================================

  gemini_AAPL_seed2024:
  Rank  Feature                  Perm Imp.   Group
  ----  -------                  ---------   -----
   1.   cash                     0.003526    cash_price_holdings
   2.   cci_30                   0.001380    tech_indicators
   3.   holdings                 0.001276    cash_price_holdings
   4.   boll_ub                  0.000284    tech_indicators
   5.   close_30_sma             0.000183    tech_indicators
   6.   close_price              0.000182    cash_price_holdings
   7.   close_60_sma             0.000175    tech_indicators
   8.   boll_lb                  0.000137    tech_indicators
   9.   rsi_30                   0.000108    tech_indicators
  10.   dx_30                    0.000097    tech_indicators
  11.   macd                     0.000050    tech_indicators
  12.   investor_confidence      0.000020    nlp_features
  13.   price_impact_potential   0.000019    nlp_features
  14.   news_relevance           0.000019    nlp_features
  15.   trend_direction          0.000018    nlp_features
  16.   sentiment                0.000017    nlp_features
  17.   earnings_impact          0.000013    nlp_features
  18.   risk_profile_change      0.000009    nlp_features

4. GROUP-LEVEL COMPARISON
==========================

  Permutation Importance (functional impact on decisions):
  ┌────────────────────────┬──────────────┬───────────┐
  │ Group                  │ Avg Imp.     │ Share     │
  ├────────────────────────┼──────────────┼───────────┤
  │ cash_price_holdings    │   0.001662   │  83.9%    │
  │ tech_indicators        │   0.000302   │  15.3%    │
  │ nlp_features           │   0.000016   │   0.8%    │
  └────────────────────────┴──────────────┴───────────┘
  Expected NLP share if uniform: 38.9% (7/18 features)
  Actual NLP share: 0.8%
  → NLP features are used at only 2% of their expected proportion

  Weight-Based Importance (MISLEADING - reflects initialization):
  ┌────────────────────────┬──────────────┬───────────┐
  │ Group                  │ Avg Weight   │ Share     │
  ├────────────────────────┼──────────────┼───────────┤
  │ cash_price_holdings    │   0.1456     │  34.2%    │
  │ tech_indicators        │   0.1400     │  32.8%    │
  │ nlp_features           │   0.1409     │  33.0%    │
  └────────────────────────┴──────────────┴───────────┘
  → Appears uniform, but this is due to orthogonal initialization
    and does NOT reflect functional importance.

5. ROOT CAUSE: Feature Scale Mismatch
======================================

  ┌────────────────────────┬──────────────────┬──────────────────┐
  │ Feature Group          │ Value Range      │ Typical Scale    │
  ├────────────────────────┼──────────────────┼──────────────────┤
  │ cash                   │ ~100,000         │ 100,000          │
  │ close_price            │ 206 - 258        │ ~230             │
  │ tech_indicators        │ -228 to +430     │ ~80 (avg range)  │
  │ nlp_features           │ -2 to +2         │ ~2.9 (avg range) │
  └────────────────────────┴──────────────────┴──────────────────┘

  Scale ratio: Tech indicators / NLP features = 28x
  Scale ratio: Cash / NLP features = ~35,000x

  The raw state vector fed to the policy network is:
  [100000, 217, 0, 1.5, 235, 212, 56, -13, 11, 219, 206, 2, 1, 1, 1, 1, 1, 0]
    ↑cash  ↑price  ↑hld  ↑--- tech indicators --------↑  ↑-- nlp features --↑

  The NLP features (range [-2, 2]) are completely dwarfed by cash
  (100,000) and price-level features (~200). Without normalization,
  the neural network's gradient updates are dominated by the
  large-scale features, making it nearly impossible for the policy
  to learn from NLP signals.

6. CROSS-MODEL CONSISTENCY
===========================

  ┌─────────────────────────┬────────────┬───────────┐
  │ Model                   │ NLP Share  │ Verdict   │
  ├─────────────────────────┼────────────┼───────────┤
  │ gemini_AAPL_seed2024    │   0.83%    │ BARELY    │
  │ gemini_MSFT_seed2024    │   0.20%    │ BARELY    │
  │ no_nlp_AAPL_seed2024    │   0.00%    │ BARELY    │
  └─────────────────────────┴────────────┴───────────┘

  The gemini models with NLP data show near-zero NLP importance,
  essentially identical to the no_nlp baseline where NLP features
  are all zeros. This confirms that having real NLP data in the
  state space makes virtually no difference to the policy's
  decisions.

7. CONCLUSION
=============

  FINDING: NLP features are NOT meaningfully used by the PPO policy.

  Despite occupying 39% of the state space (7 of 18 dimensions),
  NLP features contribute less than 1% of the policy's decision-
  making variance. The policy has effectively learned to ignore them.

  Root Cause: Severe feature scale mismatch. Cash (~100,000) and
  price-level features (~200) dominate the network's gradients,
  while NLP features (range [-2, 2]) are too small to influence
  learning. The environment does not normalize the state vector.

  Implications:
  1. Performance differences between gemini/qwen/no_nlp models are
     likely due to random seed variation, not NLP signal quality.
  2. Any apparent benefit of NLP features in the aggregate results
     is not driven by the policy actively using NLP information.
  3. To make NLP features useful, the state should be normalized
     (e.g., standard scaling or min-max normalization) before
     being fed to the policy network.

8. FILES CREATED
=================
  - feature_importance_analysis.py   (single-model analysis)
  - feature_importance_multi.py      (multi-model analysis)
  - feature_importance_results.json  (single-model results)
  - feature_importance_multi_results.json  (multi-model results)
""")
