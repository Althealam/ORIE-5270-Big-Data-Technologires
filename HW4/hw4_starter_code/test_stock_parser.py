#!/usr/bin/env python3
"""
Test script for Problem 2: Stock Price Parser
"""
import pandas as pd
from stock_parser import scrape_historical_prices, parallel_count_consecutive_decreases

def test_part1_web_scraping():
    """Test Part 1: Web Scraping"""
    print("=" * 60)
    print("Testing Part 1: Web Scraping")
    print("=" * 60)

    # Read the sample HTML file
    with open('stock_sample.html', 'r') as f:
        html_text = f.read()

    # Test the scraping function
    df = scrape_historical_prices(html_text)

    print(f"\n✓ Successfully scraped {len(df)} rows")
    print(f"✓ Columns: {list(df.columns)}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nData types:")
    print(df.dtypes)

    # Verify the data
    assert len(df) > 0, "DataFrame should not be empty"
    assert list(df.columns) == ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"], \
        "Column names don't match expected format"

    # Check that numeric columns are actually numeric
    numeric_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for col in numeric_cols:
        assert pd.api.types.is_numeric_dtype(df[col]), f"Column {col} should be numeric"

    print("\n✓ All Part 1 tests passed!")
    return df


def test_part2_mapreduce(df):
    """Test Part 2: MapReduce Analysis"""
    print("\n" + "=" * 60)
    print("Testing Part 2: Parallel MapReduce")
    print("=" * 60)

    # Test with different k values
    test_cases = [
        {"k": 2, "description": "2 consecutive decreases"},
        {"k": 3, "description": "3 consecutive decreases"},
        {"k": 4, "description": "4 consecutive decreases"},
    ]

    for test in test_cases:
        k = test["k"]
        desc = test["description"]

        try:
            count = parallel_count_consecutive_decreases(df, k=k, n_workers=2)
            print(f"\n✓ k={k} ({desc}): Found {count} occurrences")
        except Exception as e:
            print(f"\n✗ k={k} failed with error: {e}")
            raise

    print("\n✓ All Part 2 tests passed!")


def test_simple_example():
    """Test with a simple manual example"""
    print("\n" + "=" * 60)
    print("Testing with Simple Example")
    print("=" * 60)

    # Create a simple test case: 10, 9, 8, 7, 6
    # For k=3, this should have 3 occurrences: (10,9,8), (9,8,7), (8,7,6)
    test_data = {
        "Date": ["2026-03-01", "2026-03-02", "2026-03-03", "2026-03-04", "2026-03-05"],
        "Close": [10.0, 9.0, 8.0, 7.0, 6.0],
        "Open": [10.0, 9.0, 8.0, 7.0, 6.0],
        "High": [10.0, 9.0, 8.0, 7.0, 6.0],
        "Low": [10.0, 9.0, 8.0, 7.0, 6.0],
        "Adj Close": [10.0, 9.0, 8.0, 7.0, 6.0],
        "Volume": [1000, 1000, 1000, 1000, 1000],
    }
    df_test = pd.DataFrame(test_data)

    print("\nTest data (prices should be: 10, 9, 8, 7, 6):")
    print(df_test[["Date", "Close"]])

    # For k=3, expected: (10,9,8), (9,8,7), (8,7,6) = 3 occurrences
    count = parallel_count_consecutive_decreases(df_test, k=3, n_workers=2)
    print(f"\nFor k=3, found {count} occurrences (expected: 3)")

    if count == 3:
        print("✓ Simple test passed!")
    else:
        print(f"✗ Simple test failed! Expected 3 but got {count}")
        print("This might indicate an issue with the MapReduce logic.")


def main():
    print("\n" + "=" * 60)
    print("PROBLEM 2: STOCK PRICE PARSER - TEST SUITE")
    print("=" * 60)

    try:
        # Test Part 1
        df = test_part1_web_scraping()

        # Test Part 2 with real data
        test_part2_mapreduce(df)

        # Test with simple example
        test_simple_example()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        print("\nYour implementation looks good!")
        print("Make sure to submit both stock_parser.py and mr_count_consecutive_decreases.py")

    except Exception as e:
        print("\n" + "=" * 60)
        print("TEST FAILED ✗")
        print("=" * 60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
