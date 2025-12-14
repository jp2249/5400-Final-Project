#!/usr/bin/env python
# coding: utf-8

import requests
from bs4 import BeautifulSoup
import os
import pandas as pd
import re
import argparse
import sys
import logging

# Configure logger for the module
logger = logging.getLogger(__name__)

class TexasDataProcessor:
    """
    A class to scrape and process Texas executed offenders' last statements.
    """
    def __init__(self, output_dir="data/raw_data"):
        self.domain = "https://www.tdcj.texas.gov"
        self.main_url = "https://www.tdcj.texas.gov/death_row/dr_executed_offenders.html"
        self.output_dir = output_dir
        self.output_path = os.path.join(self.output_dir, "texas_last_statements_labeled.csv")

    def get_last_statement(self, url):
        """Fetches and extracts the last statement from a given URL."""
        try:
            response = requests.get(url)
            if response.status_code == 404:
                return "Page not found"
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            content_body = soup.find(id='content_right')
            if not content_body:
                 content_body = soup.find(id='body')

            if content_body:
                paragraphs = content_body.find_all('p')
                statement_text = []
                recording = False
                
                for p in paragraphs:
                    text = p.get_text(strip=True)
                    if "Last Statement:" in text:
                        parts = text.split("Last Statement:", 1)
                        if len(parts) > 1 and parts[1].strip():
                            statement_text.append(parts[1].strip())
                        recording = True
                    elif recording:
                        # Stop if we hit other sections usually at the bottom
                        if "Date of Execution:" in text or "Offender:" in text:
                            continue
                        statement_text.append(text)
                
                return " ".join(statement_text)

            return "Content body not found"

        except Exception as e:
            return f"Error fetching statement: {e}"

    def scrape_data(self):
        """Scrapes the main table and iterates through offenders to get last statements."""
        logger.info(f"Fetching main page: {self.main_url}")
        try:
            response = requests.get(self.main_url)
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to fetch main page: {e}")
            raise e

        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.find('table')
        
        if not table:
            logger.error("Could not find the table.")
            return pd.DataFrame()

        rows = table.find_all('tr')
        data_to_save = []
        logger.info(f"Found {len(rows)} rows. Processing...")

        for i, row in enumerate(rows[1:]): # Skip header
            cols = row.find_all('td')
            if len(cols) < 6:
                continue
                
            try:
                last_name = cols[3].get_text(strip=True)
                first_name = cols[4].get_text(strip=True)
                tdcj_number = cols[5].get_text(strip=True)
                
                last_statement_link = None
                link_col = cols[2]
                a_tag = link_col.find('a')
                if a_tag:
                    href = a_tag.get('href')
                    if href:
                        if href.startswith('/'):
                            last_statement_link = f"{self.domain}{href}"
                        elif href.startswith('http'):
                            last_statement_link = href
                        else:
                            last_statement_link = f"{self.domain}/death_row/{href}"
                
                statement = "N/A"
                if last_statement_link:
                    statement = self.get_last_statement(last_statement_link)
                
                data_to_save.append({
                    "TDCJ Number": tdcj_number,
                    "First Name": first_name,
                    "Last Name": last_name,
                    "Last Statement URL": last_statement_link,
                    "Last Statement": statement,
                    "is_criminal": 1,
                    "is_expected": 1,
                    "is_religious": 0
                })
                
                if i > 0 and i % 50 == 0:
                    logger.info(f"Processed {i} rows...")
                    
            except Exception as e:
                logger.warning(f"Error parsing row {i}: {e}")
                continue

        df = pd.DataFrame(data_to_save)
        logger.info(f"Collected {len(df)} records.")
        return df

    def clean_statement(self, text):
        """Cleans the 'Last Statement' column."""
        if pd.isna(text):
            return None
        
        text_str = str(text).strip()
        
        no_statement_patterns = [
            "No last statement given", "No", "No, last statement given",
            "Content body not found", "No last statement.",
            "No last statement given.", "No statement given.",
            "Page not found", "None", "None.", "N/A",
            "This inmate declined to make a last statement", 
            "this inmate inmate decline to make a last stattment", 
        ]
        
        for pattern in no_statement_patterns:
            if text_str.lower().startswith(pattern.lower()):
                return None
        
        return text_str

    def remove_context_phrases(self, text):
        """Removes specific context phrases from the start or body of the statement."""
        if pd.isna(text) or text is None:
            return text
            
        phrases = [
            r"\(written statement\)", r"Spoken:", r"written:",
            r"Verbal statement:", r"\(Spanish\)", r"Statement to the Media:",
            r"High Flight \(aviation poem\)",
            r"\(First two or three words not understood\.\)",
            r"1 Corinthians 12:31B – 13:13 \(NIV\)",
            r"1 Corinthians 12:31B – 13:13 \(NIV\) ",
            r"I would just...\(speaking in French\)",
            r"He spoke in Irish, translating to",
            r"English:", r"\(Mumbled\.\)"
        ]
        
        pattern = "|".join(phrases)
        cleaned_text = re.sub(pattern, "", str(text), flags=re.IGNORECASE)
        return cleaned_text.strip()

    def detect_religious(self, text):
        """Detects if the last statement contains religious keywords."""
        if text is None:
            return 0
            
        keywords = [
            'god', 'lord', 'jesus', 'christ', 'allah', 'holy', 'pray', 'prayer', 
            'heaven', 'bible', 'scripture', 'amen', 'bless', 'faith', 'salvation', 
            'redemption', 'psalm', 'islam', 'muslim', 'christian'
        ]
        
        text_lower = str(text).lower()
        for keyword in keywords:
            if keyword in text_lower:
                return 1
        return 0

    def process_and_save(self):
        """Runs the full scraping and cleaning pipeline."""
        logger.info("Starting data collection...")
        df = self.scrape_data()
        
        if df.empty:
            logger.warning("No data collected.")
            return

        logger.info("Cleaning 'Last Statement' column...")
        logger.info("Removing context phrases...")
        df['Last Statement'] = df['Last Statement'].apply(self.remove_context_phrases)

        logger.info("Checking for empty statements...")
        df['Last Statement'] = df['Last Statement'].apply(self.clean_statement)

        logger.info("Creating 'is_religious' column...")
        df['is_religious'] = df['Last Statement'].apply(self.detect_religious)

        logger.info(f"Saving cleaned data to: {self.output_path}")
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        df.to_csv(self.output_path, index=False)

        # Verification stats
        total = len(df)
        religious_count = df['is_religious'].sum()
        none_count = df['Last Statement'].isna().sum()

        logger.info("-" * 30)
        logger.info("Processing Complete.")
        logger.info(f"Total records: {total}")
        logger.info(f"Records with No Statement (None): {none_count}")
        logger.info(f"Records marked as Religious: {religious_count}")
        logger.info("-" * 30)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description="Scrape and process Texas Death Row Last Statements")
    parser.add_argument("--output", default="data/raw_data", help="Output directory for CSV files")
    args = parser.parse_args()

    processor = TexasDataProcessor(output_dir=args.output)
    processor.process_and_save()
