#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import wikipediaapi
import re
import argparse
import os
import sys
import logging

# Configure logger for the module
logger = logging.getLogger(__name__)

class WikipediaLastWordsScraper:
    """
    A class to scrape 'List of last words' pages from Wikipedia and parse them into a DataFrame.
    """
    def __init__(self, output_dir="data/raw_data"):
        self.wiki = wikipediaapi.Wikipedia(
            user_agent='Tyler/DSAN5400_project',
            language='en'
        )
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def parse_text(self, text):
        """Parses the raw text sections from Wikipedia into structured entries."""
        entries = []

        def parse_entry(entry_text):
            lines = [l.strip() for l in entry_text.split('\n') if l.strip()]

            if len(lines) < 2:
                return None
            
            rest = ' '.join(lines[1:]).lstrip('—-–').strip()
            date = re.search(r'\(([^)]*(?:\d{4}|c\.\s*\d+|(?:\d+th|c\.\s*\d+)\s*century)(?:\s*(?:BC|AD))?[^)]*)\)', rest)
            name = rest.split(',')[0]

            if date:
                before_date = rest[:date.start()].strip()
                title = before_date.split(',', 1)[1].strip() if ',' in before_date else ""
                context = rest[date.end():].strip(', .')
            else:
                title = rest.split(',', 1)[1].strip() if ',' in rest else ""
                context = ""
            
            return {
                'name': name,
                'title': title,
                'quote': lines[0].strip('"''"'),
                'date': date.group(1) if date else "",
                'context': context
            }

        blocks = re.split(r'\n(?=[""""])', text)

        for block in blocks:
            if block.strip():
                parsed = parse_entry(block)
                if parsed:
                    entries.append(parsed)

        return pd.DataFrame(entries)

    def scrape_century(self, century_name, page_title, limit_sections=None):
        """Scrapes a specific century page."""
        logger.info(f"Scraping {century_name}...")
        page = self.wiki.page(page_title)
        
        if not page.exists():
            logger.warning(f"Page '{page_title}' does not exist.")
            return

        text = ""
        sections_to_scrape = page.sections[:limit_sections] if limit_sections else page.sections
        for section in sections_to_scrape:
            text += section.text
        
        df = self.parse_text(text)
        logger.info(f"Collected {len(df)} records for {century_name}.")
        
        filename = f"last_words_{century_name.replace(' ', '_').lower()}.csv"
        path = os.path.join(self.output_dir, filename)
        df.to_csv(path, index=False)
        logger.info(f"Saved to {path}")

    def scrape_others(self):
        """Scrapes the main list page for pre-17th century, ironic, and notable last words."""
        logger.info("Scraping Other sections...")
        page = self.wiki.page('List of last words')
        
        if not page.exists():
            logger.warning("Main 'List of last words' page not found.")
            return

        # Pre 17th Century
        chronological_section = page.sections[0]
        text_pre5_to_17 = ""
        for subsection in chronological_section.sections[:5]:
            text_pre5_to_17 += subsection.text
        
        df_pre = self.parse_text(text_pre5_to_17)
        self._save_df(df_pre, "last_words_pre5_to_17_century.csv")

        # Ironic
        if len(page.sections) > 1:
            text_ironic = page.sections[1].text
            df_ironic = self.parse_text(text_ironic)
            self._save_df(df_ironic, "last_words_ironic.csv")

        # Notable
        if len(page.sections) > 2:
            text_notable = page.sections[2].text
            df_notable = self.parse_text(text_notable)
            self._save_df(df_notable, "last_words_notable.csv")

    def _save_df(self, df, filename):
        path = os.path.join(self.output_dir, filename)
        df.to_csv(path, index=False)
        logger.info(f"Saved {len(df)} records to {path}")

    def run(self):
        """Executes all scraping tasks."""
        self.scrape_century("21st Century", 'List of last words (21st century)', limit_sections=3)
        self.scrape_century("20th Century", 'List of last words (20th century)', limit_sections=10)
        self.scrape_century("19th Century", 'List of last words (19th century)', limit_sections=10)
        self.scrape_century("18th Century", 'List of last words (18th century)', limit_sections=10)
        self.scrape_others()
        logger.info("Done!")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(description="Scrape Wikipedia for Last Words")
    parser.add_argument("--output", default="data/raw_data", help="Output directory")
    args = parser.parse_args()

    scraper = WikipediaLastWordsScraper(output_dir=args.output)
    scraper.run()
