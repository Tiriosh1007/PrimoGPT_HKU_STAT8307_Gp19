#!/usr/bin/env python3
"""
Feature Importance Analysis for PPO Trading Models
===================================================
Determines whether NLP features are actually used by the policy network.

Approach:
1. Retrain PPO model on AAPL data (same config as impact test)
2. Policy weight analysis: absolute weights of first layer
3. Permutation importance: permute each feature and measure action change
4. Group comparison: NLP features vs tech indicators vs cash/price/holdings

State space (18 dims for single stock):
  [0]  cash
  [1]  close price
  [2]  num_stock_shares (holdings)
  [3]  macd
  [4]  boll_ub
  [5]  boll_lb
  [6]  rsi_30
  [7]  cci_30
  [8]  dx_30
  [9]  close_30_sma
  [10] close_60_sma
  [11] news_relevance
  [12] sentiment
  [13] price_impact_potential
  [14] trend_direction
  [15] earnings_impact
  [16] investor_confidence
  [17] risk_profile_change
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ── Paths ──
OUR_CODE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(OUR_CODE)
PRIMO_GPT = os.path.join(PROJECT, "PrimoGPT-main")

sys.path.insert(0, PRIMO_GPT)
sys.path.insert(0, OUR_CODE)
sys.path.insert(0, os.path.join(OUR_CODE, "ppo_training"))

# ── Feature Names ──
FEATURE_NAMES = [
    "cash",                    # idx 0
    "close_price",             # idx 1
    "holdings",                # idx 2
    "macd",                    # idx 3
    "boll_ub",                 # idx 4
    "boll_lb",                 # idx 5
    "rsi_30",                  # idx 6
    "cci_30",                  # idx 7
    "dx_30",                   # idx 8
    "close_30_sma",            # idx 9
    "close_60_sma",            # idx 10
    "news_relevance",          # idx 11
    "sentiment",               # idx 12
    "price_impact_potential",  # idx 13
    "trend_direction",         # idx 14
    "earnings_impact",         # idx 15
    "investor_confidence",     # idx 16
    "risk_profile_change",     # idx 17
]

FEATURE_GROUPS = {
    "cash_price_holdings": [0, 1, 2],
    "tech_indicators": [3, 4, 5, 6, 7, 8, 9, 10],
    "nlp_features": [11, 12, 13, 14, 15, 16, 17],
}

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

TRAIN_START = "2022-04-01"
TRAIN_END = "2024-07-31"
TEST_START = "2024-08-01"
TEST_END = "2025-02-28"

PPO_PARAMS = dict(
    n_steps=2048,
    ent_coef=0.01,
    learning_rate=0.00025,
    batch_size=128,
)

TRANSACTION_COST = 0.001
TOTAL_TIMESTEPS = 400000  # Same as original training
SEED = 2024  # Use the seed that had 42 trades


def load_nlp_data(llm_config, ticker):
    """Load NLP signal data."""
    if llm_config == "no_nlp":
        return None

    if llm_config == "gemini":
        path = os.path.join(OUR_CODE, "ppo_data", f"gemini_ppo_{ticker}_data.csv")
    elif llm_config.startswith("qwen"):
        subdir = "LLM_data"
        path = os.path.join(OUR_CODE, subdir, f"{llm_config}_ppo_{ticker}_data.csv")
    else:
        raise ValueError(f"Unknown LLM config: {llm_config}")

    if not os.path.exists(path):
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
    """Load price data using FixedYahooDownloader + FeatureEngineer."""
    from fixed_yahoodownloader import FixedYahooDownloader
    from finrl.meta.preprocessor.preprocessors import FeatureEngineer

    df = FixedYahooDownloader(
        start_date=TRAIN_START,
        end_date=TEST_END,
        ticker_list=[ticker]
    ).fetch_data()

    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=["macd", "boll_ub", "boll_lb", "rsi_30", "cci_30", "dx_30", "close_30_sma", "close_60_sma"],
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
    from finrl.meta.preprocessor.preprocessors import data_split

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
            elif col.endswith("_y") and col[:-2] + "_x" not in result.columns:
                result = result.rename(columns={col: col[:-2]})
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


def train_model(llm_config, ticker, seed):
    """Train PPO model and return (model, test_env, test_df, train_df)."""
    from finrl.meta.env_primo_trading.env_primorl import StockTradingEnv
    from finrl.agents.stablebaselines3.models import DRLAgent
    from stable_baselines3.common.logger import configure

    print(f"\n{'='*60}")
    print(f"  Training PPO: {llm_config}_{ticker}_seed{seed}")
    print(f"  Timesteps: {TOTAL_TIMESTEPS}")
    print(f"{'='*60}")

    nlp_df = load_nlp_data(llm_config, ticker)
    price_df = load_price_data(ticker)
    train_df, test_df, full_df = prepare_combined_data(price_df, nlp_df, llm_config, ticker)

    stock_dim = 1
    state_space = 1 + 2 * stock_dim + 8 * stock_dim + len(FUNDAMENTAL_INDICATORS) * stock_dim

    env_kwargs = dict(
        hmax=1000,
        initial_amount=100000,
        buy_cost_pct=[TRANSACTION_COST],
        sell_cost_pct=[TRANSACTION_COST],
        num_stock_shares=[0],
        stock_dim=stock_dim,
        state_space=state_space,
        action_space=stock_dim,
        tech_indicator_list=["macd", "boll_ub", "boll_lb", "rsi_30", "cci_30", "dx_30", "close_30_sma", "close_60_sma"],
        fundamental_indicator_list=FUNDAMENTAL_INDICATORS,
        reward_type="diff_sharpe",
        reward_scaling=1,
        initial=True,
        print_verbosity=999999,
    )

    e_train_gym = StockTradingEnv(df=train_df, **env_kwargs)
    agent = DRLAgent(env=e_train_gym)
    model = agent.get_model("ppo", model_kwargs=PPO_PARAMS, seed=seed)

    # Suppress logger
    import tempfile
    tmp = tempfile.mkdtemp()
    new_logger = configure(tmp, ["csv"])
    model.set_logger(new_logger)

    import time
    t0 = time.time()
    trained_model = agent.train_model(model=model, tb_log_name="ppo", total_timesteps=TOTAL_TIMESTEPS)
    print(f"  Training time: {time.time()-t0:.0f}s")

    # Create test env
    e_test_gym = StockTradingEnv(df=test_df, **env_kwargs)

    return trained_model, e_test_gym, test_df, train_df


def collect_observations(env, model, n_steps=None):
    """Collect observations from the test environment using the trained model."""
    reset_result = env.reset()
    obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    
    observations = [obs.copy()]
    actions = []
    
    if n_steps is None:
        n_steps = len(env.df.index.unique()) - 1
    
    for i in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
        action = np.clip(action, -1.0, 1.0)
        actions.append(action[0])
        
        step_result = env.step(action)
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            obs, reward, done, info = step_result
        
        observations.append(obs.copy())
        if done:
            break
    
    return np.array(observations), np.array(actions)


# ================================================================
# ANALYSIS METHOD 1: Policy Network Weight Analysis
# ================================================================
def analyze_policy_weights(model):
    """Analyze the first-layer weights of the policy network."""
    import torch
    
    policy = model.policy
    
    print("\n" + "="*70)
    print("  METHOD 1: Policy Network Weight Analysis")
    print("="*70)
    
    # Get all parameter names and shapes
    print("\nPolicy network architecture:")
    for name, param in policy.named_parameters():
        print(f"  {name}: shape={param.shape}")
    
    # Find the first linear layer of the policy (actor) network
    # In SB3 PPO with MlpPolicy, the actor network is typically:
    #   policy.latent_pi (shared features extractor -> latent)
    #   or policy.mlp_extractor.policy_net (if no shared features)
    
    # Try different paths to find the input layer
    first_layer_weights = None
    first_layer_name = None
    
    # Approach 1: Look for features_extractor
    if hasattr(policy, 'features_extractor') and policy.features_extractor is not None:
        for name, param in policy.features_extractor.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                first_layer_weights = param.detach().cpu().numpy()
                first_layer_name = f"features_extractor.{name}"
                break
    
    # Approach 2: Look for latent_pi (shared layers)
    if first_layer_weights is None and hasattr(policy, 'mlp_extractor'):
        if hasattr(policy.mlp_extractor, 'shared_net'):
            for name, param in policy.mlp_extractor.shared_net.named_parameters():
                if 'weight' in name and param.dim() >= 2:
                    first_layer_weights = param.detach().cpu().numpy()
                    first_layer_name = f"mlp_extractor.shared_net.{name}"
                    break
        # Approach 3: Look for policy_net (separate actor)
        if first_layer_weights is None and hasattr(policy.mlp_extractor, 'policy_net'):
            for name, param in policy.mlp_extractor.policy_net.named_parameters():
                if 'weight' in name and param.dim() >= 2:
                    first_layer_weights = param.detach().cpu().numpy()
                    first_layer_name = f"mlp_extractor.policy_net.{name}"
                    break
    
    # Approach 4: Try latent_pi directly
    if first_layer_weights is None and hasattr(policy, 'latent_pi'):
        for name, param in policy.latent_pi.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                first_layer_weights = param.detach().cpu().numpy()
                first_layer_name = f"latent_pi.{name}"
                break
    
    # Approach 5: Just scan all parameters
    if first_layer_weights is None:
        for name, param in policy.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                # Check if input dim matches state space
                if param.shape[-1] == 18 or param.shape[0] == 18:
                    first_layer_weights = param.detach().cpu().numpy()
                    first_layer_name = name
                    break
                # Take the first one that could be input layer
                if first_layer_weights is None:
                    first_layer_weights = param.detach().cpu().numpy()
                    first_layer_name = name
    
    if first_layer_weights is None:
        print("  ERROR: Could not find first layer weights!")
        return None
    
    print(f"\nFirst layer: {first_layer_name}, shape: {first_layer_weights.shape}")
    
    # Compute feature importance from weights
    # For a linear layer y = Wx + b, importance of input feature i = sum of |W[:, i]| across output neurons
    # This measures how much each input feature contributes to the layer's output
    if first_layer_weights.ndim == 2:
        # Weight matrix shape: (output_dim, input_dim) in PyTorch convention
        # Or we need to check which dimension is input
        if first_layer_weights.shape[1] == 18:
            # Standard: (out, in) - columns are input features
            abs_weights = np.abs(first_layer_weights)
            feature_importance = np.mean(abs_weights, axis=0)  # average across output neurons
        elif first_layer_weights.shape[0] == 18:
            # Transposed: (in, out)
            abs_weights = np.abs(first_layer_weights)
            feature_importance = np.mean(abs_weights, axis=1)
        else:
            print(f"  WARNING: Weight matrix shape {first_layer_weights.shape} doesn't match 18-dim state")
            # Use column-wise if wider, row-wise if taller
            if first_layer_weights.shape[1] < first_layer_weights.shape[0]:
                feature_importance = np.mean(np.abs(first_layer_weights), axis=0)
            else:
                feature_importance = np.mean(np.abs(first_layer_weights), axis=1)
    else:
        print(f"  WARNING: Unexpected weight dimensions: {first_layer_weights.ndim}")
        return None
    
    return feature_importance, first_layer_name, first_layer_weights


# ================================================================
# ANALYSIS METHOD 2: Permutation Feature Importance
# ================================================================
def permutation_importance(model, observations, baseline_actions=None, n_repeats=20):
    """Compute permutation feature importance.
    
    For each feature, permute that feature across the observation batch
    and measure how much the predicted actions change.
    Larger change = more important feature.
    """
    import torch
    
    print("\n" + "="*70)
    print("  METHOD 2: Permutation Feature Importance")
    print("="*70)
    
    n_obs = len(observations)
    n_features = observations.shape[1]
    
    # Get baseline predictions
    with torch.no_grad():
        obs_tensor = torch.FloatTensor(observations)
        # Use model.predict for each observation to get deterministic actions
        baseline_preds = []
        for i in range(n_obs):
            action, _ = model.predict(observations[i], deterministic=True)
            baseline_preds.append(action)
    baseline_preds = np.array(baseline_preds).squeeze()
    
    if baseline_actions is not None:
        print(f"  Using {len(observations)} observations from test environment")
        print(f"  Baseline action mean: {np.mean(baseline_preds):.6f}, std: {np.std(baseline_preds):.6f}")
    
    # Permutation importance for each feature
    importance_scores = np.zeros(n_features)
    importance_std = np.zeros(n_features)
    
    for feat_idx in range(n_features):
        score_repeats = []
        for _ in range(n_repeats):
            # Create permuted observations
            perm_obs = observations.copy()
            perm_obs[:, feat_idx] = np.random.permutation(perm_obs[:, feat_idx])
            
            # Get predictions with permuted feature
            perm_preds = []
            for i in range(n_obs):
                action, _ = model.predict(perm_obs[i], deterministic=True)
                perm_preds.append(action)
            perm_preds = np.array(perm_preds).squeeze()
            
            # Measure change: mean absolute difference
            change = np.mean(np.abs(perm_preds - baseline_preds))
            score_repeats.append(change)
        
        importance_scores[feat_idx] = np.mean(score_repeats)
        importance_std[feat_idx] = np.std(score_repeats)
    
    return importance_scores, importance_std


# ================================================================
# ANALYSIS METHOD 3: Gradient-based Feature Importance (Saliency)
# ================================================================
def gradient_importance(model, observations):
    """Compute gradient-based feature importance using input gradients.
    
    Compute the gradient of the action output w.r.t. each input feature,
    averaged across observations. This tells us how sensitive the policy
    is to small changes in each feature.
    """
    import torch
    
    print("\n" + "="*70)
    print("  METHOD 3: Gradient-based Feature Importance (Input Saliency)")
    print("="*70)
    
    n_obs = len(observations)
    n_features = observations.shape[1]
    
    # Get the policy network
    policy = model.policy
    policy.eval()
    
    grad_magnitudes = np.zeros(n_features)
    
    for i in range(n_obs):
        obs_tensor = torch.FloatTensor(observations[i]).unsqueeze(0).requires_grad_(True)
        
        # Forward pass through policy
        with torch.enable_grad():
            # Get the action distribution
            latent_pi = policy.mlp_extractor.forward_policy(obs_tensor)
            # Get the action
            distribution = policy._get_action_dist_from_latent(latent_pi)
            # Take the mean action
            action = distribution.deterministic_sample()
        
        # Backward pass
        if action.requires_grad:
            action.backward(torch.ones_like(action))
        
        if obs_tensor.grad is not None:
            grad_magnitudes += np.abs(obs_tensor.grad.numpy().squeeze())
    
    grad_magnitudes /= n_obs
    
    return grad_magnitudes


# ================================================================
# Main Analysis
# ================================================================
def main():
    import torch
    
    llm_config = "gemini"
    ticker = "AAPL"
    seed = SEED
    
    print("="*70)
    print("  FEATURE IMPORTANCE ANALYSIS FOR PPO TRADING MODELS")
    print("  Testing whether NLP features are actually used by the policy")
    print("="*70)
    
    # Train model
    model, test_env, test_df, train_df = train_model(llm_config, ticker, seed)
    
    # Collect test observations
    print("\nCollecting test observations...")
    observations, actions = collect_observations(test_env, model)
    print(f"  Collected {len(observations)} observations, {len(actions)} actions")
    print(f"  Action range: [{actions.min():.4f}, {actions.max():.4f}]")
    print(f"  Non-zero actions: {np.sum(actions != 0)}/{len(actions)}")
    
    # ---- METHOD 1: Policy Weight Analysis ----
    weight_result = analyze_policy_weights(model)
    
    if weight_result is not None:
        feature_importance_weights, layer_name, weight_matrix = weight_result
        
        print(f"\n  Feature Importance from {layer_name}:")
        print(f"  {'Feature':<25} {'Weight Importance':>18} {'Group':>20}")
        print(f"  {'-'*25} {'-'*18} {'-'*20}")
        
        for group_name, indices in FEATURE_GROUPS.items():
            for idx in indices:
                print(f"  {FEATURE_NAMES[idx]:<25} {feature_importance_weights[idx]:>18.6f} {group_name:>20}")
        
        # Group averages
        print(f"\n  Group Average Weight Importance:")
        group_avgs = {}
        for group_name, indices in FEATURE_GROUPS.items():
            avg = np.mean(feature_importance_weights[indices])
            group_avgs[group_name] = avg
            print(f"    {group_name:<25} {avg:.6f}")
        
        # Relative importance (normalized)
        total = sum(group_avgs.values())
        print(f"\n  Relative Group Importance (weight-based):")
        for group_name, avg in sorted(group_avgs.items(), key=lambda x: -x[1]):
            pct = avg / total * 100
            print(f"    {group_name:<25} {pct:6.2f}%")
    
    # ---- METHOD 2: Permutation Importance ----
    perm_scores, perm_std = permutation_importance(model, observations, actions, n_repeats=20)
    
    print(f"\n  Permutation Feature Importance (20 repeats):")
    print(f"  {'Feature':<25} {'Perm Importance':>16} {'Std':>10} {'Group':>20}")
    print(f"  {'-'*25} {'-'*16} {'-'*10} {'-'*20}")
    
    for group_name, indices in FEATURE_GROUPS.items():
        for idx in indices:
            print(f"  {FEATURE_NAMES[idx]:<25} {perm_scores[idx]:>16.6f} {perm_std[idx]:>10.6f} {group_name:>20}")
    
    # Group averages
    print(f"\n  Group Average Permutation Importance:")
    group_avgs_perm = {}
    for group_name, indices in FEATURE_GROUPS.items():
        avg = np.mean(perm_scores[indices])
        group_avgs_perm[group_name] = avg
        print(f"    {group_name:<25} {avg:.6f}")
    
    total_perm = sum(group_avgs_perm.values())
    print(f"\n  Relative Group Importance (permutation-based):")
    for group_name, avg in sorted(group_avgs_perm.items(), key=lambda x: -x[1]):
        pct = avg / total_perm * 100 if total_perm > 0 else 0
        print(f"    {group_name:<25} {pct:6.2f}%")
    
    # ---- METHOD 3: Gradient-based Importance ----
    try:
        grad_scores = gradient_importance(model, observations)
        
        print(f"\n  Gradient-based Feature Importance:")
        print(f"  {'Feature':<25} {'Grad Magnitude':>16} {'Group':>20}")
        print(f"  {'-'*25} {'-'*16} {'-'*20}")
        
        for group_name, indices in FEATURE_GROUPS.items():
            for idx in indices:
                print(f"  {FEATURE_NAMES[idx]:<25} {grad_scores[idx]:>16.6f} {group_name:>20}")
        
        print(f"\n  Group Average Gradient Importance:")
        group_avgs_grad = {}
        for group_name, indices in FEATURE_GROUPS.items():
            avg = np.mean(grad_scores[indices])
            group_avgs_grad[group_name] = avg
            print(f"    {group_name:<25} {avg:.6f}")
        
        total_grad = sum(group_avgs_grad.values())
        print(f"\n  Relative Group Importance (gradient-based):")
        for group_name, avg in sorted(group_avgs_grad.items(), key=lambda x: -x[1]):
            pct = avg / total_grad * 100 if total_grad > 0 else 0
            print(f"    {group_name:<25} {pct:6.2f}%")
    except Exception as e:
        print(f"\n  Gradient analysis failed: {e}")
        grad_scores = None
    
    # ---- COMBINED SUMMARY ----
    print("\n" + "="*70)
    print("  COMBINED SUMMARY: Are NLP features used by the policy?")
    print("="*70)
    
    # Rank all features by each method
    print(f"\n  Feature Rankings (1=most important):")
    print(f"  {'Feature':<25} {'Weight Rank':>12} {'Perm Rank':>10} {'Group':>20}")
    print(f"  {'-'*25} {'-'*12} {'-'*10} {'-'*20}")
    
    if weight_result is not None:
        weight_ranks = np.argsort(-feature_importance_weights) + 1  # descending
    else:
        weight_ranks = np.zeros(18, dtype=int)
    
    perm_ranks = np.argsort(-perm_scores) + 1
    
    for group_name, indices in FEATURE_GROUPS.items():
        for idx in indices:
            print(f"  {FEATURE_NAMES[idx]:<25} {weight_ranks[idx]:>12} {perm_ranks[idx]:>10} {group_name:>20}")
    
    # NLP vs Non-NLP comparison
    nlp_indices = FEATURE_GROUPS["nlp_features"]
    non_nlp_indices = FEATURE_GROUPS["cash_price_holdings"] + FEATURE_GROUPS["tech_indicators"]
    
    print(f"\n  NLP vs Non-NLP Comparison:")
    if weight_result is not None:
        nlp_weight = np.mean(feature_importance_weights[nlp_indices])
        non_nlp_weight = np.mean(feature_importance_weights[non_nlp_indices])
        print(f"    Weight-based:  NLP avg = {nlp_weight:.6f}, Non-NLP avg = {non_nlp_weight:.6f}, ratio = {nlp_weight/non_nlp_weight:.3f}")
    
    nlp_perm = np.mean(perm_scores[nlp_indices])
    non_nlp_perm = np.mean(perm_scores[non_nlp_indices])
    print(f"    Perm-based:    NLP avg = {nlp_perm:.6f}, Non-NLP avg = {non_nlp_perm:.6f}, ratio = {nlp_perm/non_nlp_perm:.3f}")
    
    if grad_scores is not None:
        nlp_grad = np.mean(grad_scores[nlp_indices])
        non_nlp_grad = np.mean(grad_scores[non_nlp_indices])
        print(f"    Gradient-based: NLP avg = {nlp_grad:.6f}, Non-NLP avg = {non_nlp_grad:.6f}, ratio = {nlp_grad/non_nlp_grad:.3f}")
    
    # Specific NLP feature rankings
    print(f"\n  NLP Feature Rankings (permutation importance):")
    nlp_perm_scores = [(FEATURE_NAMES[i], perm_scores[i]) for i in nlp_indices]
    nlp_perm_scores.sort(key=lambda x: -x[1])
    for rank, (name, score) in enumerate(nlp_perm_scores, 1):
        print(f"    {rank}. {name:<25} {score:.6f}")
    
    # Save results
    results = {
        "weight_importance": {FEATURE_NAMES[i]: float(feature_importance_weights[i]) for i in range(18)} if weight_result else None,
        "permutation_importance": {FEATURE_NAMES[i]: float(perm_scores[i]) for i in range(18)},
        "permutation_std": {FEATURE_NAMES[i]: float(perm_std[i]) for i in range(18)},
        "gradient_importance": {FEATURE_NAMES[i]: float(grad_scores[i]) for i in range(18)} if grad_scores is not None else None,
        "group_summary": {
            "weight_based": {k: float(np.mean(feature_importance_weights[v])) for k, v in FEATURE_GROUPS.items()} if weight_result else None,
            "permutation_based": {k: float(np.mean(perm_scores[v])) for k, v in FEATURE_GROUPS.items()},
            "gradient_based": {k: float(np.mean(grad_scores[v])) for k, v in FEATURE_GROUPS.items()} if grad_scores is not None else None,
        },
        "config": f"{llm_config}_{ticker}_seed{seed}",
    }
    
    output_path = os.path.join(OUR_CODE, "feature_importance_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {output_path}")
    
    # Final verdict
    print("\n" + "="*70)
    print("  VERDICT")
    print("="*70)
    
    if weight_result:
        nlp_pct_weight = group_avgs["nlp_features"] / total * 100
        print(f"  Weight-based:   NLP features account for {nlp_pct_weight:.1f}% of first-layer importance")
    
    nlp_pct_perm = group_avgs_perm["nlp_features"] / total_perm * 100
    print(f"  Perm-based:     NLP features account for {nlp_pct_perm:.1f}% of importance")
    
    if grad_scores is not None:
        nlp_pct_grad = group_avgs_grad["nlp_features"] / total_grad * 100
        print(f"  Gradient-based: NLP features account for {nlp_pct_grad:.1f}% of importance")
    
    # Expected NLP share if uniform: 7/18 = 38.9%
    expected_nlp_pct = 7/18 * 100
    print(f"\n  Expected NLP share if uniform: {expected_nlp_pct:.1f}%")
    
    if weight_result:
        if nlp_pct_weight < expected_nlp_pct * 0.5:
            print(f"  -> Weight analysis: NLP features are UNDER-REPRESENTED (< 50% of expected)")
        elif nlp_pct_weight > expected_nlp_pct * 1.5:
            print(f"  -> Weight analysis: NLP features are OVER-REPRESENTED (> 150% of expected)")
        else:
            print(f"  -> Weight analysis: NLP features are used PROPORTIONALLY")
    
    if nlp_pct_perm < expected_nlp_pct * 0.5:
        print(f"  -> Permutation analysis: NLP features are UNDER-REPRESENTED (< 50% of expected)")
    elif nlp_pct_perm > expected_nlp_pct * 1.5:
        print(f"  -> Permutation analysis: NLP features are OVER-REPRESENTED (> 150% of expected)")
    else:
        print(f"  -> Permutation analysis: NLP features are used PROPORTIONALLY")
    
    print("="*70)


if __name__ == "__main__":
    main()
