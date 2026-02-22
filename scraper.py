"""
scraper.py - Scrape houses-for-sale listings from ikman.lk

Extracts listing data from the embedded JSON (window.initialData) on each
search results page, paginating through all pages until the target number
of records is reached.

Usage:
    python scraper.py                  # Scrape 5,500 records (default)
    python scraper.py --pages 3        # Scrape only the first 3 pages (~75 records)
    python scraper.py --total 1000     # Scrape 1,000 records
"""

from __future__ import annotations

import argparse
import json
import re
import time
import random
import sys

import requests
from bs4 import BeautifulSoup

from store_data import append_to_csv, save_to_csv, get_record_count

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
BASE_URL = "https://ikman.lk/en/ads/sri-lanka/houses-for-sale"
LISTINGS_PER_PAGE = 25
DEFAULT_TOTAL_RECORDS = 5500
CSV_FILENAME = "houses_for_sale.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}

MAX_RETRIES = 3
RETRY_BACKOFF = 2  # seconds (multiplied by attempt number)


# ──────────────────────────────────────────────
# Parsing helpers
# ──────────────────────────────────────────────
def extract_json_data(html: str) -> list[dict] | None:
    """
    Extract the listing data from the embedded window.initialData JSON
    in the page HTML.

    Returns a list of ad dicts or None if extraction fails.
    """
    # Try to find the initialData script block
    # Pattern: window.initialData = {...};
    match = re.search(
        r'window\.initialData\s*=\s*(\{.*?\})\s*;?\s*</script>',
        html,
        re.DOTALL,
    )
    if not match:
        # Fallback: try to find it in a different format
        match = re.search(
            r'window\.initialData\s*=\s*(\{.*?\})\s*;',
            html,
            re.DOTALL,
        )
    if not match:
        return None

    try:
        data = json.loads(match.group(1))
        # Navigate the JSON structure to get the ads list
        ads = data.get("serp", {}).get("ads", {}).get("data", {}).get("ads", [])
        return ads if ads else None
    except (json.JSONDecodeError, AttributeError):
        return None


def parse_details(details_str: str) -> tuple[str, str]:
    """
    Parse the 'details' string like 'Bedrooms: 4, Bathrooms: 3'
    into separate bedroom and bathroom values.

    Returns:
        Tuple of (bedrooms, bathrooms) as strings.
    """
    bedrooms = ""
    bathrooms = ""

    if not details_str:
        return bedrooms, bathrooms

    bed_match = re.search(r"Bedrooms?:\s*(\d+)", details_str)
    bath_match = re.search(r"Bathrooms?:\s*(\d+)", details_str)

    if bed_match:
        bedrooms = bed_match.group(1)
    if bath_match:
        bathrooms = bath_match.group(1)

    return bedrooms, bathrooms


def parse_ad(ad: dict) -> dict:
    """
    Transform a raw ad dict from the JSON data into our standardized format.
    """
    bedrooms, bathrooms = parse_details(ad.get("details", ""))

    return {
        "title": (ad.get("title") or "").strip(),
        "price": (ad.get("price") or "").strip(),
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "location": (ad.get("location") or "").strip(),
        "posted_time": (ad.get("timeAgo") or ad.get("postedTime") or "").strip(),
        "ad_url": f"https://ikman.lk/en/ad/{ad.get('slug', '')}",
    }


# ──────────────────────────────────────────────
# HTML fallback parser (if JSON extraction fails)
# ──────────────────────────────────────────────
def parse_listings_from_html(html: str) -> list[dict]:
    """
    Fallback: Parse listing data directly from the HTML DOM using
    BeautifulSoup if JSON extraction fails.
    """
    soup = BeautifulSoup(html, "lxml")
    listings = []

    # Find all listing cards
    cards = soup.select("li.normal--29mBy a.card-link--3ssYv")
    if not cards:
        # Try alternative selectors
        cards = soup.select('a[href*="/en/ad/"]')

    for card in cards:
        href = card.get("href", "")
        if "/en/ad/" not in href:
            continue

        title_el = card.select_one("h2")
        price_el = card.select_one('[class*="price"]')
        details_el = card.select_one('[class*="description"]')
        location_el = card.select_one('[class*="updated-time"], [class*="subtitle"]')

        title = title_el.get_text(strip=True) if title_el else ""
        price = price_el.get_text(strip=True) if price_el else ""
        details = details_el.get_text(strip=True) if details_el else ""
        location_text = location_el.get_text(strip=True) if location_el else ""

        # Extract location (first part before comma)
        location = location_text.split(",")[0].strip() if location_text else ""

        bedrooms, bathrooms = parse_details(details)

        ad_url = f"https://ikman.lk{href}" if href.startswith("/") else href

        listings.append({
            "title": title,
            "price": price,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "location": location,
            "posted_time": "",
            "ad_url": ad_url,
        })

    return listings


# ──────────────────────────────────────────────
# Page fetcher
# ──────────────────────────────────────────────
def fetch_page(page_num: int, session: requests.Session) -> str | None:
    """
    Fetch a single listing page with retry logic.

    Returns the HTML string or None on failure.
    """
    url = f"{BASE_URL}?page={page_num}" if page_num > 1 else BASE_URL

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(url, headers=HEADERS, timeout=30)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            wait_time = RETRY_BACKOFF * attempt + random.uniform(0, 1)
            if attempt < MAX_RETRIES:
                print(f"  ⚠ Page {page_num} attempt {attempt} failed: {e}")
                print(f"    Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                print(f"  ✗ Page {page_num} FAILED after {MAX_RETRIES} attempts: {e}")
                return None

    return None


# ──────────────────────────────────────────────
# Scrape one page
# ──────────────────────────────────────────────
def scrape_listing_page(page_num: int, session: requests.Session) -> list[dict]:
    """
    Scrape a single listing page and return parsed records.
    Uses JSON extraction first, falls back to HTML parsing.
    """
    html = fetch_page(page_num, session)
    if html is None:
        return []

    # Primary method: extract from embedded JSON
    ads = extract_json_data(html)
    if ads:
        return [parse_ad(ad) for ad in ads]

    # Fallback: parse HTML directly
    print(f"  ℹ Page {page_num}: JSON extraction failed, using HTML fallback")
    return parse_listings_from_html(html)


# ──────────────────────────────────────────────
# Main scraping orchestrator
# ──────────────────────────────────────────────
def scrape_all(
    total_records: int = DEFAULT_TOTAL_RECORDS,
    max_pages: int | None = None,
    incremental: bool = True,
) -> list[dict]:
    """
    Scrape listing pages until the target number of records is collected.

    Args:
        total_records: Target number of records to collect.
        max_pages: If set, stop after this many pages regardless of total.
        incremental: If True, append to CSV after each page (crash resilient).

    Returns:
        Complete list of all scraped records.
    """
    pages_needed = max_pages or (total_records // LISTINGS_PER_PAGE + 1)
    all_records: list[dict] = []
    failed_pages: list[int] = []

    print(f"🏠 ikman.lk Houses-for-Sale Scraper")
    print(f"{'─' * 50}")
    print(f"  Target records : {total_records}")
    print(f"  Pages to scrape: {pages_needed}")
    print(f"  Incremental CSV: {'Yes' if incremental else 'No'}")
    print(f"{'─' * 50}\n")

    session = requests.Session()

    # If incremental and file exists, check how many records we already have
    if incremental:
        existing = get_record_count(CSV_FILENAME)
        if existing > 0:
            print(f"📂 Found {existing} existing records in {CSV_FILENAME}")
            start_page = (existing // LISTINGS_PER_PAGE) + 1
            print(f"   Resuming from page {start_page}\n")
        else:
            start_page = 1
    else:
        start_page = 1

    for page in range(start_page, pages_needed + 1):
        # Progress indicator
        if page % 10 == 1 or page == start_page:
            print(f"📄 Scraping pages {page}-{min(page + 9, pages_needed)} "
                  f"of {pages_needed}...")

        records = scrape_listing_page(page, session)

        if not records:
            failed_pages.append(page)
            print(f"  ✗ Page {page}: no records extracted")
        else:
            all_records.extend(records)
            if incremental:
                append_to_csv(records, CSV_FILENAME)

            if page % 10 == 0:
                print(f"  ✓ Progress: {len(all_records)} records collected "
                      f"({page}/{pages_needed} pages)")

        # Check if we've reached the target
        if len(all_records) >= total_records:
            print(f"\n🎯 Reached target of {total_records} records!")
            all_records = all_records[:total_records]
            break

        # Polite delay between requests (1-2 seconds)
        delay = 1.0 + random.uniform(0, 1.0)
        time.sleep(delay)

    # Summary
    print(f"\n{'═' * 50}")
    print(f"✅ Scraping complete!")
    print(f"   Total records : {len(all_records)}")
    print(f"   Failed pages  : {len(failed_pages)}")
    if failed_pages:
        print(f"   Failed page #s: {failed_pages[:20]}{'...' if len(failed_pages) > 20 else ''}")
    print(f"{'═' * 50}")

    return all_records


# ──────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Scrape houses-for-sale listings from ikman.lk"
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=None,
        help="Number of pages to scrape (overrides --total)",
    )
    parser.add_argument(
        "--total",
        type=int,
        default=DEFAULT_TOTAL_RECORDS,
        help=f"Target number of records to collect (default: {DEFAULT_TOTAL_RECORDS})",
    )
    parser.add_argument(
        "--no-incremental",
        action="store_true",
        help="Disable incremental saving (save all at end instead)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start fresh, ignoring any existing CSV data",
    )
    args = parser.parse_args()

    incremental = not args.no_incremental

    # If --fresh, remove existing CSV
    if args.fresh:
        import os
        csv_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), CSV_FILENAME
        )
        if os.path.isfile(csv_path):
            os.remove(csv_path)
            print(f"🗑 Removed existing {CSV_FILENAME}")

    records = scrape_all(
        total_records=args.total,
        max_pages=args.pages,
        incremental=incremental,
    )

    # If not incremental, save all at the end
    if not incremental and records:
        save_to_csv(records, CSV_FILENAME)

    # Final CSV record count
    final_count = get_record_count(CSV_FILENAME)
    print(f"\n📊 Final CSV contains {final_count} records")


if __name__ == "__main__":
    main()
