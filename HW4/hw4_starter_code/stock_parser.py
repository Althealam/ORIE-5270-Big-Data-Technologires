import os
import json
import math
import tempfile
from bs4 import BeautifulSoup
import pandas as pd
from mr_count_consecutive_decreases import MRCountConsecutiveDecreases

CANONICAL_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]

def scrape_historical_prices(html_text: str) -> pd.DataFrame:
    """
    Return a pandas DataFrame with columns:
    Date, Open, High, Low, Close, Adj Close, Volume
    """
    # 1. get html by using beautifulsoup
    soup = BeautifulSoup(html_text, 'html.parser')

    # 2. find the historical prices table: same page has other cbd04 tables (headlines,
    #    etc.). The first cbd04 table without c1a9f may not be the 7-column price table,
    #    so we take the candidate that yields the most valid data rows.
    def rows_from_price_table(table):
        tbody = table.find('tbody')
        if tbody is None:
            return []
        out = []
        for row in tbody.find_all('tr'):
            if row.find("td", attrs={"colspan": True}):
                continue
            cells = row.find_all("td")
            if len(cells) == 7:
                out.append([c.get_text(strip=True) for c in cells])
        return out

    data = []
    for t in soup.find_all("table"):
        classes = t.get("class", [])
        if "cbd04" not in classes or "c1a9f" in classes:
            continue
        candidate = rows_from_price_table(t)
        if len(candidate) > len(data):
            data = candidate

    if not data:
        raise ValueError("Could not find the historical prices table")

    # 3. Create DataFrame
    df = pd.DataFrame(data, columns=CANONICAL_COLUMNS)

    # 4. Convert numerical columns to numeric values
    numeric_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for col in numeric_cols:
        # Remove commas and convert to float
        df[col] = df[col].str.replace(',', '', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

def split_into_chunks(prices: list, n_workers: int) -> list:
    """
    Split the prices list into n_workers chunks.
    Each chunk should have roughly equal size.
    """
    n = len(prices)
    chunk_size = math.ceil(n / n_workers)

    chunks = []
    for i in range(0, n, chunk_size):
        chunk = prices[i:i + chunk_size]
        chunks.append(chunk)

    return chunks

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
