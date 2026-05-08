import pandas as pd
import numpy as np
import os
import glob

def evaluate_predictions(base_dir="our_code/ppo_results"):
    # This matches Stage 2 Downstream constraints expected by the thesis.
    metrics = []

    # Get all models running
    models = os.listdir(base_dir)
    for model in models:
        model_path = os.path.join(base_dir, model)
        if not os.path.isdir(model_path):
            continue
            
        tickers = os.listdir(model_path)
        for ticker in tickers:
            ticker_path = os.path.join(model_path, ticker, "predictions", "account_value.csv")
            
            if not os.path.exists(ticker_path):
                continue
                
            df = pd.read_csv(ticker_path)
            
            # Assuming 'account_value' and 'date' columns exist based on FinRL defaults
            if len(df) == 0:
                continue
                
            start_value = df['account_value'].iloc[0]
            end_value = df['account_value'].iloc[-1]
            
            # Cumulative Return
            cumulative_return = (end_value - start_value) / start_value
            
            # Daily Returns
            df['daily_return'] = df['account_value'].pct_change()
            
            # Annualized Volatility (assuming 252 trading days)
            annual_volatility = df['daily_return'].std() * np.sqrt(252)
            
            # Sharpe Ratio
            if df['daily_return'].std() != 0:
                sharpe_ratio = df['daily_return'].mean() / df['daily_return'].std() * np.sqrt(252)
            else:
                sharpe_ratio = 0

            # Max Drawdown
            rolling_max = df['account_value'].cummax()
            drawdown = df['account_value'] / rolling_max - 1.0
            max_drawdown = drawdown.min()
            
            metrics.append({
                "Model": model,
                "Ticker": ticker,
                "Cumulative Return (%)": round(cumulative_return * 100, 2),
                "Sharpe Ratio": round(sharpe_ratio, 2),
                "Annual Volatility (%)": round(annual_volatility * 100, 2),
                "Max Drawdown (%)": round(max_drawdown * 100, 2)
            })

    results_df = pd.DataFrame(metrics)
    print("\n================== STAGE 2: PPO TRADING PERFORMANCE RESULTS ==================")
    print(results_df.to_string(index=False))
    
    # Check if we should calculate portfolio averages like the paper
    for model in results_df['Model'].unique():
        model_data = results_df[results_df['Model'] == model]
        print(f"\nAverage Performance for {model}:")
        print(f"Mean Cumulative Return: {model_data['Cumulative Return (%)'].mean():.2f}%")
        print(f"Mean Sharpe Ratio: {model_data['Sharpe Ratio'].mean():.2f}")

def print_stage1_metrics(base_dir="our_code/benchmarks_output"):
    print("\n================== STAGE 1: NLP SIGNAL EXTRACTION PERFORMANCE ==================")
    metrics_list = []
    
    # Map the directories to tickers to make printing nice
    dir_to_ticker = {
        "outputs_full_run": "AAPL",
        "outputs_AMZN_full_run": "AMZN",
        "outputs_CRM_full_run": "CRM",
        "outputs_MSFT_full_run": "MSFT",
        "outputs_NFLX_full_run": "NFLX"
    }
    
    import json
    for dirname, ticker in dir_to_ticker.items():
        metrics_path = os.path.join(base_dir, dirname, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                data = json.load(f)
                
            metrics_list.append({
                "Model": "gemini",
                "Ticker": ticker,
                "News Rel": round(data['field_accuracy']['news_relevance']*100, 1),
                "Sentiment": round(data['field_accuracy']['sentiment']*100, 1),
                "Px Impact": round(data['field_accuracy']['price_impact_potential']*100, 1),
                "Trend": round(data['field_accuracy']['trend_direction']*100, 1),
                "Earn Impact": round(data['field_accuracy']['earnings_impact']*100, 1),
                "Inv Conf": round(data['field_accuracy']['investor_confidence']*100, 1),
                "Risk": round(data['field_accuracy']['risk_profile_change']*100, 1),
                "Exact Match": round(data['exact_match_accuracy']*100, 1)
            })
            
    if metrics_list:
        metrics_df = pd.DataFrame(metrics_list)
        print(metrics_df.to_string(index=False))
        print("\nMean Exact Match (Gemini):", round(metrics_df['Exact Match'].mean(), 2), "%")
        
        print("\nFeature Averages (Gemini):")
        for col in ["News Rel", "Sentiment", "Px Impact", "Trend", "Earn Impact", "Inv Conf", "Risk"]:
            print(f"  {col}: {round(metrics_df[col].mean(), 2)}%")

if __name__ == '__main__':
    print_stage1_metrics()
    evaluate_predictions()
