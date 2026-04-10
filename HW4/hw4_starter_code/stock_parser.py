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

    # 2. find the correct table 
    table = None
    tables = soup.find_all('table')
    for t in tables:
        classes = t.get('class', [])
        # We want tables with 'cbd04' but without 'c1a9f' (hidden)
        if 'cbd04' in classes and 'c1a9f' not in classes:
            table = t
            break

    if table is None:
        raise ValueError("Could not find the historical prices table")

    # 3. Extract table rows from tbody
    tbody = table.find('tbody')
    if tbody is None:
        raise ValueError("Could not find tbody in the table")

    rows = tbody.find_all('tr')

    # 4. Extract data from rows
    data = []
    for row in rows:
        # Skip rows with colspan (these are ads/announcements)
        if row.find('td', attrs={'colspan': True}):
            continue

        # Get all cells in the row
        cells = row.find_all('td')

        # We expect exactly 7 columns
        if len(cells) == 7:
            # Extract text from each cell
            row_data = [cell.get_text(strip=True) for cell in cells]
            data.append(row_data)

    # 5. Create DataFrame
    df = pd.DataFrame(data, columns=CANONICAL_COLUMNS)

    # 6. Convert numerical columns to numeric values
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
