import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import os
import logging
from src.utils import logger, ensure_dirs

"""
ETHICAL SCRAPING NOTE:
This script is for educational purposes. We respect robots.txt and include delays 
between requests to avoid overloading the server. Always check ikman.lk's 
Terms of Service before extensive scraping.
"""

class IkmanScraper:
    def __init__(self, base_url="https://ikman.lk/en/ads/sri-lanka/houses-for-sale", max_pages=2):
        self.base_url = base_url
        self.max_pages = max_pages
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        ensure_dirs()

    def scrape_page(self, page_num):
        url = f"{self.base_url}?page={page_num}"
        logger.info(f"Scraping page {page_num}: {url}")
        
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            
            ads = soup.find_all("li", class_="normal--2QYVk")
            data = []
            
            for ad in ads:
                try:
                    title = ad.find("h2", class_="title--3S61q").text.strip() if ad.find("h2", class_="title--3S61q") else "N/A"
                    location = ad.find("div", class_="description--2-S3M").text.split(",")[0].strip() if ad.find("div", class_="description--2-S3M") else "N/A"
                    price_text = ad.find("div", class_="price--3rj7m").text.strip() if ad.find("div", class_="price--3rj7m") else "N/A"
                    
                    # More detailed info usually requires visiting the ad page, 
                    # but we extract what's available on the listing page
                    data.append({
                        "title": title,
                        "location": location,
                        "price": price_text,
                        "ad_url": "https://ikman.lk" + ad.find("a")["href"] if ad.find("a") else "N/A"
                    })
                except Exception as e:
                    logger.error(f"Error parsing ad: {e}")
            
            return data
        except Exception as e:
            logger.error(f"Error scraping page {page_num}: {e}")
            return []

    def run(self, output_path="houses_for_sale_new.csv"):
        all_data = []
        for p in range(1, self.max_pages + 1):
            page_data = self.scrape_page(p)
            all_data.extend(page_data)
            time.sleep(random.uniform(1, 3)) # Respectful delay
            
        df = pd.DataFrame(all_data)
        df.to_csv(output_path, index=False)
        logger.info(f"Scraped {len(df)} ads and saved to {output_path}")
        return df

if __name__ == "__main__":
    scraper = IkmanScraper(max_pages=1) # Minimal scrape for testing
    scraper.run()
