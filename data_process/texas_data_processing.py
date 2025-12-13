#!/usr/bin/env python
# coding: utf-8

# # Texas Death Row Data Processing
# 
# This notebook combines the functionality of scraping Texas executed offenders' last statements and cleaning the collected data.
# 
# ## Parts
# 1. **Data Collection**: Scrapes data from the Texas Department of Criminal Justice website.
# 2. **Data Cleaning**: Processes the text of the last statements to remove unwanted context and flag religious content.

# In[31]:


import requests
from bs4 import BeautifulSoup
import csv
import os
import time
import pandas as pd
import re


# ## Part 1: Data Collection

# In[32]:


DOMAIN = "https://www.tdcj.texas.gov"
MAIN_URL = "https://www.tdcj.texas.gov/death_row/dr_executed_offenders.html"

def get_last_statement(url):
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


# In[33]:


print(f"Fetching main page: {MAIN_URL}")
try:
    response = requests.get(MAIN_URL)
    response.raise_for_status()
except Exception as e:
    print(f"Failed to fetch main page: {e}")
    # Stop execution if main page fails, practically raising error here for notebook flow
    raise e

soup = BeautifulSoup(response.content, 'html.parser')

table = soup.find('table')
if not table:
    print("Could not find the table.")
else:
    rows = table.find_all('tr')
    data_to_save = []
    print(f"Found {len(rows)} rows. Processing...")

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
                        last_statement_link = f"{DOMAIN}{href}"
                    elif href.startswith('http'):
                        last_statement_link = href
                    else:
                        # Relative to /death_row/
                        last_statement_link = f"{DOMAIN}/death_row/{href}"
            
            statement = "N/A"
            if last_statement_link:
                statement = get_last_statement(last_statement_link)
            
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
            
            if i % 10 == 0:
                print(f"Processed {i} rows...")
                # Optional: break for testing
                # if i > 20: break 
                
        except Exception as e:
            print(f"Error parsing row {i}: {e}")
            continue

    # Convert to DataFrame in memory instead of saving intermediate CSV
    df = pd.DataFrame(data_to_save)
    print(f"Collected {len(df)} records.")


# ## Part 2: Data Cleaning

# In[34]:


def clean_statement(text):
    """
    Cleans the 'Last Statement' column by setting specific 'no statement' phrases to None.
    Also handles NaN/float values.
    """
    if pd.isna(text):
        return None
    
    # Normalize text for checking
    text_str = str(text).strip()
    
    # Exact phrases or startswith patterns to identify no statement
    no_statement_patterns = [
        "No last statement given",
        "No",
        "No, last statement given",
        "Content body not found",
        "No last statement.",
        "No last statement given.",
        "No statement given.",
        "Page not found",
        "None",
        "None.",
        "N/A",
        "This inmate declined to make a last statement", 
        "this inmate inmate decline to make a last stattment", 
    ]
    
    for pattern in no_statement_patterns:
        # Case insensitive check
        if text_str.lower().startswith(pattern.lower()):
            return None
            
    return text_str

def remove_context_phrases(text):
    """
    Removes specific context phrases from the start or body of the statement,
    maintaining the rest of the sentence.
    """
    if pd.isna(text) or text is None:
        return text
        
    # Phrases to remove
    phrases = [
        r"\(written statement\)",
        r"Spoken:",
        r"written:",
        r"Verbal statement:",
        r"\(Spanish\)",
        r"Statement to the Media:",
        r"High Flight \(aviation poem\)",
        r"\(First two or three words not understood\.\)",
        r"1 Corinthians 12:31B – 13:13 \(NIV\)",
        r"1 Corinthians 12:31B – 13:13 \(NIV\) ",
        r"I would just...\(speaking in French\)",
        r"He spoke in Irish, translating to",
        r"English:",
        r"\(Mumbled\.\)"
    ]
    
    # Join into a single pattern
    pattern = "|".join(phrases)
    
    # Replace with empty string, preserving the rest
    cleaned_text = re.sub(pattern, "", str(text), flags=re.IGNORECASE)
    
    return cleaned_text.strip()


# In[35]:


def detect_religious(text):
    """
    Detects if the last statement contains religious keywords.
    Returns True if found, False otherwise.
    """
    if text is None:
        return 0
        
    keywords = [
        'god', 'lord', 'jesus', 'christ', 'allah', 'holy', 'pray', 'prayer', 
        'heaven', 'bible', 'scripture', 'amen', 'bless', 'faith', 'salvation', 
        'redemption', 'psalm', 'islam', 'muslim', 'christian'
    ]
    
    text_lower = text.lower()
    
    for keyword in keywords:
        if keyword in text_lower:
            return 1
            
    return 0


# In[36]:


# File paths
output_path = os.path.join("data/raw_data", "texas_last_statements_labeled.csv")

# Check if df exists (from Part 1)
if 'df' not in locals():
    print("Error: DataFrame 'df' not found. Run Part 1 first.")
else:
    print("Cleaning 'Last Statement' column...")

    # Apply removal of context phrases FIRST
    print("Removing context phrases...")
    df['Last Statement'] = df['Last Statement'].apply(remove_context_phrases)

    # Apply cleaning (no statement checks)
    df['Last Statement'] = df['Last Statement'].apply(clean_statement)

    print("Creating 'is_religious' column...")
    # Apply religious detection
    df['is_religious'] = df['Last Statement'].apply(detect_religious)

    print(f"Saving cleaned data to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    # Verification stats
    total = len(df)
    religious_count = df['is_religious'].sum()
    none_count = df['Last Statement'].isna().sum()

    print("-" * 30)
    print(f"Processing Complete.")
    print(f"Total records: {total}")
    print(f"Records with No Statement (None): {none_count}")
    print(f"Records marked as Religious: {religious_count}")
    print("-" * 30)

