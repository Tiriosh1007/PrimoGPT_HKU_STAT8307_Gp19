import pandas as pd
import json
import os
import glob

def ingest_gemini_baseline():
    """
    Transforms the pre-run Gemini JSONL outputs into the PPO training data format.
    """
    source_dirs = {
        "AAPL": "benchmarks_output/outputs_full_run",
        "AMZN": "benchmarks_output/outputs_AMZN_full_run",
        "CRM": "benchmarks_output/outputs_CRM_full_run",
        "MSFT": "benchmarks_output/outputs_MSFT_full_run",
        "NFLX": "benchmarks_output/outputs_NFLX_full_run",
    }
    
    # Original columns needed: 
    # Date,Adj Close Price,Returns,Bin Label,News Relevance,Sentiment,Price Impact Potential,Trend Direction,Earnings Impact,Investor Confidence,Risk Profile Change,Prompt

    for ticker, dir_path in source_dirs.items():
        base_data_path = f"PrimoGPT-main/notebooks/6. PrimoRL trading with NLP features/data/{ticker}_data.csv"
        predictions_path = f"our_code/{dir_path}/predictions.jsonl"
        
        if not os.path.exists(base_data_path) or not os.path.exists(predictions_path):
            print(f"Skipping {ticker}: Missing base or predictions file.")
            continue
            
        print(f"Processing Gemini baseline for {ticker}...")
        df = pd.read_csv(base_data_path)
        
        preds = []
        with open(predictions_path, 'r') as f:
            for line in f:
                preds.append(json.loads(line))
        
        # Create mapping of row_id to prediction dictionary
        pred_map = {p['row_id']: p['prediction'] for p in preds}
        
        # Apply predictions directly to columns based on target mapping in market_sentiment_eval
        for i in range(len(df)):
            if i in pred_map:
                p = pred_map[i]
                df.at[i, 'News Relevance'] = p.get('news_relevance', 0)
                df.at[i, 'Sentiment'] = p.get('sentiment', 0)
                df.at[i, 'Price Impact Potential'] = p.get('price_impact_potential', 0)
                df.at[i, 'Trend Direction'] = p.get('trend_direction', 0)
                df.at[i, 'Earnings Impact'] = p.get('earnings_impact', 0)
                df.at[i, 'Investor Confidence'] = p.get('investor_confidence', 0)
                df.at[i, 'Risk Profile Change'] = p.get('risk_profile_change', 0)
                
        out_path = f"our_code/ppo_data/gemini_ppo_{ticker}_data.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved merged Gemini data to {out_path}")

if __name__ == '__main__':
    ingest_gemini_baseline()
