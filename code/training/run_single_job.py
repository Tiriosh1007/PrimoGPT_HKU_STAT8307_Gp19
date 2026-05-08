#!/usr/bin/env python3
"""
Run a single training job (one algo, one ticker, one seed).
Used by the parallel launcher.
"""
import sys
import os

# LOCAL PrimoGPT first
_PRIMOGPT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../PrimoGPT-main'))
if _PRIMOGPT_PATH not in sys.path:
    sys.path.insert(0, _PRIMOGPT_PATH)

import pandas as pd
import numpy as np
import json
import argparse
from datetime import datetime
import torch

from fixed_yahoodownloader import FixedYahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_primo_trading.env_primorl import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

# Constants
DATA_DOWNLOAD_START = '2022-04-01'
TRAIN_START_DATE = '2022-04-01'
TRAIN_END_DATE = '2024-07-31'
TRADE_START_DATE = '2024-08-01'
TRADE_END_DATE = '2025-02-28'
INDICATORS = ['macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30',
              'dx_30', 'close_30_sma', 'close_60_sma']
FUNDAMENTAL_INDICATORS = ['news_relevance', 'sentiment', 'price_impact_potential',
    'trend_direction', 'earnings_impact', 'investor_confidence', 'risk_profile_change']
COLUMN_MAPPING = {
    'News Relevance': 'news_relevance', 'Sentiment': 'sentiment',
    'Price Impact Potential': 'price_impact_potential', 'Trend Direction': 'trend_direction',
    'Earnings Impact': 'earnings_impact', 'Investor Confidence': 'investor_confidence',
    'Risk Profile Change': 'risk_profile_change'
}
INITIAL_AMOUNT = 100000


class SafeTrainingCallback(BaseCallback):
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
                    print(f"  NaN in {name} at step {self.num_timesteps}")
                    return False
        return True


def load_data(ticker, data_path):
    df_nlp = pd.read_csv(data_path)
    df_nlp = df_nlp.rename(columns={'Date': 'date'})
    if 'Adj Close Price' in df_nlp.columns:
        df_nlp = df_nlp.rename(columns={'Adj Close Price': 'close_nlp'})
    elif 'Close' in df_nlp.columns:
        df_nlp = df_nlp.rename(columns={'Close': 'close_nlp'})
    df_nlp = df_nlp.rename(columns=COLUMN_MAPPING)
    cols_to_drop = [c for c in ['Adj Close Price', 'Returns', 'Bin Label', 'Prompt', 'close_nlp'] if c in df_nlp.columns]
    df_nlp = df_nlp.drop(columns=cols_to_drop, errors='ignore')

    df_yahoo = FixedYahooDownloader(
        start_date=DATA_DOWNLOAD_START, end_date=TRADE_END_DATE, ticker_list=[ticker]
    ).fetch_data()

    fe = FeatureEngineer(
        use_technical_indicator=True, tech_indicator_list=INDICATORS,
        use_vix=True, use_turbulence=False, user_defined_feature=False
    )
    processed = fe.preprocess_data(df_yahoo)

    processed['date'] = pd.to_datetime(processed['date']).dt.strftime('%Y-%m-%d')
    df_nlp['date'] = pd.to_datetime(df_nlp['date']).dt.strftime('%Y-%m-%d')
    processed_full = processed.merge(df_nlp, on='date', how='left')

    for col in FUNDAMENTAL_INDICATORS:
        processed_full[col] = processed_full.groupby('tic')[col].shift(1)
    processed_full = processed_full.fillna(0)

    for col in ['close', 'tic']:
        if f'{col}_x' in processed_full.columns and f'{col}_y' in processed_full.columns:
            processed_full[col] = processed_full[f'{col}_x']
            processed_full = processed_full.drop(columns=[f'{col}_x', f'{col}_y'])

    train_data = data_split(processed_full, TRAIN_START_DATE, TRAIN_END_DATE)
    test_data = data_split(processed_full, TRADE_START_DATE, TRADE_END_DATE)
    return train_data, test_data


def make_env(train_data):
    stock_dimension = 1
    state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension + len(FUNDAMENTAL_INDICATORS)*stock_dimension
    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension

    env_kwargs = {
        "hmax": 1000, "initial_amount": INITIAL_AMOUNT,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list, "sell_cost_pct": sell_cost_list,
        "state_space": state_space, "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "fundamental_indicator_list": FUNDAMENTAL_INDICATORS,
        "action_space": stock_dimension, "reward_scaling": 1,
        "reward_type": "pct_return", "verbose": 0
    }
    e_train_gym = StockTradingEnv(df=train_data, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()
    return env_train, env_kwargs


def predict(model, test_data, env_kwargs):
    env_kwargs_copy = {**env_kwargs, "verbose": 0}
    e_trade_gym = StockTradingEnv(df=test_data, **env_kwargs_copy)
    test_env, test_obs = e_trade_gym.get_sb_env()
    test_env.reset()
    max_steps = len(e_trade_gym.df.index.unique()) - 1

    for i in range(len(e_trade_gym.df.index.unique())):
        action, _states = model.predict(test_obs, deterministic=True)
        if np.any(np.isnan(action)) or np.any(np.isinf(action)):
            action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
        test_obs, rewards, dones, info = test_env.step(action)
        if i == max_steps - 1:
            account_memory = test_env.env_method(method_name="save_asset_memory")
            actions_memory = test_env.env_method(method_name="save_action_memory")
        if dones[0]:
            break

    df_account = account_memory[0]
    df_actions = actions_memory[0]
    final_value = df_account.iloc[-1, 1]
    ret_pct = (final_value - INITIAL_AMOUNT) / INITIAL_AMOUNT * 100
    return ret_pct, df_account, df_actions


def run_job(algo, ticker, seed, results_dir, tune_iter=None, params_json=None):
    """Run a single training+eval job."""
    label = f"{algo}_tune{tune_iter}" if tune_iter is not None else algo
    print(f"[{label}] {ticker} seed={seed} starting...")

    data_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), f'../ppo_data/gemini_ppo_{ticker}_data.csv'))
    if not os.path.exists(data_path):
        print(f"[{label}] {ticker} data missing")
        return None

    train_data, test_data = load_data(ticker, data_path)
    env_train, env_kwargs = make_env(train_data)
    agent = DRLAgent(env=env_train)

    if algo == "sac":
        model_kwargs = {
            "batch_size": 256, "buffer_size": 100000,
            "learning_rate": 0.0001, "learning_starts": 2000,
            "ent_coef": "auto_0.1", "tau": 0.005, "gamma": 0.99,
        }
        policy_kwargs = dict(net_arch=[64, 64], n_critics=2)
        model = agent.get_model("sac", model_kwargs=model_kwargs,
                                policy_kwargs=policy_kwargs, seed=seed)
        total_timesteps = 400000

    elif algo == "td3":
        params = json.loads(params_json) if params_json else {}
        net_arch = params.pop("net_arch", [64, 64])
        # DRLAgent.get_model expects action_noise as STRING key: "normal" or "ornstein_uhlenbeck"
        # It creates the noise object internally
        action_noise_type = params.pop("action_noise", "normal")
        params["action_noise"] = action_noise_type  # pass as string, not object
        policy_kwargs = dict(net_arch=net_arch, n_critics=2)
        model = agent.get_model("td3", model_kwargs=params,
                                policy_kwargs=policy_kwargs, seed=seed)
        total_timesteps = 400000

    # Logging
    rdir = f"{results_dir}/seed{seed}/{ticker}"
    os.makedirs(rdir, exist_ok=True)
    new_logger = configure(rdir, ["csv", "tensorboard"])
    model.set_logger(new_logger)

    # Train
    safe_callback = SafeTrainingCallback(max_grad_norm=1.0, check_freq=2048)
    trained = model.learn(total_timesteps=total_timesteps,
                          tb_log_name=algo, callback=safe_callback)

    # Evaluate
    ret_pct, df_account, df_actions = predict(trained, test_data, env_kwargs)
    print(f"[{label}] {ticker} seed={seed} -> {ret_pct:+.1f}%")

    # Save
    out_dir = f"{rdir}/predictions"
    os.makedirs(out_dir, exist_ok=True)
    df_account.to_csv(f"{out_dir}/account_value.csv", index=False)
    df_actions.to_csv(f"{out_dir}/actions.csv", index=False)

    # Save quick result
    with open(f"{rdir}/result.json", 'w') as f:
        json.dump({"ticker": ticker, "seed": seed, "return_pct": ret_pct,
                    "algo": algo, "tune_iter": tune_iter}, f)

    return ret_pct


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, required=True, choices=['sac', 'td3'])
    parser.add_argument('--ticker', type=str, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--tune_iter', type=int, default=None)
    parser.add_argument('--params_json', type=str, default=None)
    args = parser.parse_args()

    ret = run_job(args.algo, args.ticker, args.seed, args.results_dir,
                  args.tune_iter, args.params_json)
    if ret is not None:
        print(f"DONE: {args.algo} {args.ticker} seed={args.seed} ret={ret:+.1f}%")
    else:
        print(f"FAILED: {args.algo} {args.ticker} seed={args.seed}")
