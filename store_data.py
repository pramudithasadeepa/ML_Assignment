"""
store_data.py - CSV storage utility for scraped house listing data.

Provides functions to save, append, and load house listing data to/from CSV files.
"""

from __future__ import annotations

import csv
import os

# Default CSV column headers
FIELDNAMES = [
    "title",
    "price",
    "bedrooms",
    "bathrooms",
    "location",
    "posted_time",
    "ad_url",
]


def save_to_csv(data: list[dict], filename: str = "houses_for_sale.csv") -> None:
    """
    Save a list of dicts to a CSV file (overwrites existing file).

    Args:
        data: List of dictionaries with house listing data.
        filename: Output CSV file path.
    """
    if not data:
        print("⚠ No data to save.")
        return

    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(data)

    print(f"✅ Saved {len(data)} records to {filepath}")


def append_to_csv(data: list[dict], filename: str = "houses_for_sale.csv") -> None:
    """
    Append a list of dicts to an existing CSV file.
    Creates the file with headers if it doesn't exist.

    Args:
        data: List of dictionaries with house listing data.
        filename: Output CSV file path.
    """
    if not data:
        return

    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    file_exists = os.path.isfile(filepath)

    with open(filepath, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerows(data)


def load_csv(filename: str = "houses_for_sale.csv") -> list[dict]:
    """
    Load CSV data back into a list of dictionaries.

    Args:
        filename: CSV file path to load.

    Returns:
        List of dictionaries with house listing data.
    """
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)

    if not os.path.isfile(filepath):
        print(f"⚠ File not found: {filepath}")
        return []

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        data = list(reader)

    print(f"📂 Loaded {len(data)} records from {filepath}")
    return data


def get_record_count(filename: str = "houses_for_sale.csv") -> int:
    """
    Get the number of records already saved in the CSV (excluding header).

    Args:
        filename: CSV file path.

    Returns:
        Number of data rows in the file, or 0 if file doesn't exist.
    """
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)

    if not os.path.isfile(filepath):
        return 0

    with open(filepath, "r", encoding="utf-8") as f:
        # Subtract 1 for header row
        return sum(1 for _ in f) - 1
