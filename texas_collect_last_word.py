import requests
from bs4 import BeautifulSoup
import csv
import os
import time


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
                    # Stop if we hit other sections usually at the bottom (though usually they are above)
                    # But sometimes there might be footer info in paragraphs?
                    # In the example, there is nothing after.
                    if "Date of Execution:" in text or "Offender:" in text:
                        # These usually appear BEFORE the statement, so if we see them again, it's weird.
                        # But let's be safe.
                        continue
                    statement_text.append(text)
            
            return " ".join(statement_text)

        return "Content body not found"

    except Exception as e:
        return f"Error fetching statement: {e}"






def main():
    print(f"Fetching main page: {MAIN_URL}")
    try:
        response = requests.get(MAIN_URL)
        response.raise_for_status()
    except Exception as e:
        print(f"Failed to fetch main page: {e}")
        return

    soup = BeautifulSoup(response.content, 'html.parser')
    
    table = soup.find('table')
    if not table:
        print("Could not find the table.")
        return

    rows = table.find_all('tr')
    
    data_to_save = []
    
    print(f"Found {len(rows)} rows. Processing...")
    
    # Columns:
    # 0: Execution #
    # 1: Link (Inmate Info)
    # 2: Link (Last Statement)
    # 3: Last Name
    # 4: First Name
    # 5: TDCJ Number
    
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
                "crime": 1
            })
            
            if i % 10 == 0:
                print(f"Processed {i} rows...")
                
        except Exception as e:
            print(f"Error parsing row {i}: {e}")
            continue

    data_folder = "data"
    os.makedirs(data_folder, exist_ok=True)

    filename = os.path.join(data_folder, "texas_last_statements.csv")
    keys = ["TDCJ Number", "First Name", "Last Name", "Last Statement","Last Statement URL", "crime"]

    try:
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(data_to_save)
        print(f"Successfully saved {len(data_to_save)} records to {filename}")
    except Exception as e:
        print(f"Error saving to CSV: {e}")








if __name__ == "__main__":
    main()
