import os
import json
import math
import tempfile
from bs4 import BeautifulSoup
import pandas as pd

CANONICAL_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]

def scrape_historical_prices(html_text: str) -> pd.DataFrame:
    """
    Return a pandas DataFrame with columns:
    Date, Open, High, Low, Close, Adj Close, Volume
    """
    raise NotImplementedError

def parallel_count_consecutive_decreases(
    df: pd.DataFrame,
    k: int,
    n_workers: int = 4,
) -> int:
    if k <= 1:
        raise ValueError("k must be at least 2.")

    required_cols = {"Date", "Close"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    prices = df["Close"].astype(float).tolist()
    if len(prices) < k:
        return 0

    chunks = split_into_chunks(prices, n_workers)

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        input_path = f.name
        for chunk_id, chunk in enumerate(chunks):
            record = {"chunk_id": chunk_id, "prices": chunk}
            f.write(json.dumps(record) + "\n")

    try:
        job = MRCountConsecutiveDecreases(
            args=["--no-conf", "-r", "local", "--k", str(k), input_path]
        )

        with job.make_runner() as runner:
            runner.run()

            for key, value in job.parse_output(runner.cat_output()):
                if key == "total_count":
                    return int(value)

        raise RuntimeError("MRJob did not produce total_count output.")

    finally:
        if os.path.exists(input_path):
            os.remove(input_path)
