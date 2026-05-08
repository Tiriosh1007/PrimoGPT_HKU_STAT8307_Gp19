"""
Bug-free YahooDownloader: fixes the yfinance column-ordering bug.

PROBLEM with original YahooDownloader:
  yfinance >= 0.2.35 returns columns in ALPHABETICAL order:
    [Date, Adj Close, Close, High, Low, Open, Volume]
  But the original renames them POSITIONALLY as:
    [date, open, high, low, close, adjcp, volume, tic]
  This swaps: open=AdjClose, high=Close, low=High, close=Open.

FIX: Two changes:
  1. Use auto_adjust=True -> yfinance adjusts all OHLC internally,
     Close = Adj Close, no separate Adj Close column needed.
  2. Use column-NAME-based mapping instead of positional.

This produces CORRECT OHLC data (low <= close <= high always holds)
and the close price reflects total return (dividend-adjusted).
"""
from __future__ import annotations
import pandas as pd
import yfinance as yf


class FixedYahooDownloader:
    """YahooDownloader with correct column mapping and auto-adjusted prices."""

    def __init__(self, start_date: str, end_date: str, ticker_list: list):
        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list

    def fetch_data(self, proxy=None) -> pd.DataFrame:
        data_df = pd.DataFrame()
        num_failures = 0
        for tic in self.ticker_list:
            kwargs = dict(
                start=self.start_date, end=self.end_date,
                multi_level_index=False, auto_adjust=True  # KEY FIX: auto_adjust
            )
            if proxy:
                kwargs['proxy'] = proxy
            temp_df = yf.download(tic, **kwargs)
            temp_df["tic"] = tic
            if len(temp_df) > 0:
                data_df = pd.concat([data_df, temp_df], axis=0)
            else:
                num_failures += 1
        if num_failures == len(self.ticker_list):
            raise ValueError("no data is fetched.")

        data_df = data_df.reset_index()

        # ---- FIX: Column-NAME mapping (not positional) ----
        # With auto_adjust=True, columns are: Date, Close, High, Low, Open, Volume
        rename_map = {
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",  # Already adjusted (= Adj Close)
            "Volume": "volume",
        }
        data_df = data_df.rename(columns=rename_map)

        # No need for "close = adjcp; drop adjcp" since auto_adjust handles it

        data_df["day"] = data_df["date"].dt.dayofweek
        data_df["date"] = data_df.date.apply(lambda x: x.strftime("%Y-%m-%d"))
        data_df = data_df.dropna()
        data_df = data_df.reset_index(drop=True)
        print("Shape of DataFrame: ", data_df.shape)

        data_df = data_df.sort_values(by=["date", "tic"]).reset_index(drop=True)

        # Verify OHLC sanity
        violations = ((data_df["low"] > data_df["close"]) | (data_df["close"] > data_df["high"])).sum()
        if violations > 0:
            print(f"  WARNING: {violations} OHLC violations (low>close or close>high)")
        else:
            print("  OHLC sanity check PASSED: low <= close <= high for all rows")

        return data_df

    def select_equal_rows_stock(self, df):
        df_check = df.tic.value_counts()
        df_check = pd.DataFrame(df_check).reset_index()
        df_check.columns = ["tic", "counts"]
        mean_df = df_check.counts.mean()
        equal_list = list(df.tic.value_counts() >= mean_df)
        names = df.tic.value_counts().index
        select_stocks_list = list(names[equal_list])
        df = df[df.tic.isin(select_stocks_list)]
        return df
