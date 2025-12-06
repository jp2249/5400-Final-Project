import requests
from bs4 import BeautifulSoup
import csv
import os
import time
from datetime import datetime


DOMAIN = "https://www.tdcj.texas.gov"
MAIN_URL = "https://www.tdcj.texas.gov/death_row/dr_executed_offenders.html"


def format_date(date_str):
    """Convert date from MM/DD/YYYY format to 'DD Month YYYY' format"""
    try:
        # Parse the date string (MM/DD/YYYY)
        date_obj = datetime.strptime(date_str, '%m/%d/%Y')
        # Format to 'DD Month YYYY'
        return date_obj.strftime('%d %B %Y')
    except (ValueError, AttributeError):
        # Return original string if parsing fails
        return date_str


def get_last_statement(url):
    """Extract the last statement from an individual inmate page"""
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
    print(f"Starting script...")
    print(f"Fetching main page: {MAIN_URL}")
    try:
        response = requests.get(MAIN_URL)
        print(f"Response status: {response.status_code}")
        response.raise_for_status()
        print("Successfully fetched page")
    except Exception as e:
        print(f"Failed to fetch main page: {e}")
        return

    soup = BeautifulSoup(response.content, 'html.parser')
    
    table = soup.find('table')
    if not table:
        print("Could not find the table.")
        return

    rows = table.find_all('tr')
    
    # First, let's examine the header to understand the table structure
    if rows:
        header_row = rows[0]
        header_cols = header_row.find_all(['th', 'td'])
        print("Table columns:")
        for i, col in enumerate(header_cols):
            print(f"  Column {i}: {col.get_text(strip=True)}")
    
    data_to_save = []
    
    print(f"Found {len(rows)} rows. Processing...")
    
    for i, row in enumerate(rows[1:]): # Skip header
        cols = row.find_all('td')
        if len(cols) < 6:
            continue
            
        try:
            # Extract all available columns
            execution_number = cols[0].get_text(strip=True) if len(cols) > 0 else ""
            
            # Based on the header output, column 7 is "Date"
            raw_execution_date = cols[7].get_text(strip=True) if len(cols) > 7 else ""
            # Format the date to 'DD Month YYYY' format
            execution_date = format_date(raw_execution_date) if raw_execution_date else ""
            
            first_name = cols[4].get_text(strip=True) if len(cols) > 4 else ""
            last_name = cols[3].get_text(strip=True) if len(cols) > 3 else ""
            tdcj_number = cols[5].get_text(strip=True) if len(cols) > 5 else ""
            
            # Extract last statement link from column 2
            last_statement_link = None
            link_col = cols[2] if len(cols) > 2 else None
            if link_col:
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
            
            # Get the last statement
            statement = "N/A"
            if last_statement_link:
                print(f"  Fetching statement for {first_name} {last_name}...")
                statement = get_last_statement(last_statement_link)
                # Small delay to be respectful to the server
                time.sleep(0.5)
            
            data_to_save.append({
                "Execution Number": execution_number,
                "First Name": first_name,
                "Last Name": last_name,
                "TDCJ Number": tdcj_number,
                "Execution Date": execution_date,
                "Last Statement": statement,
                "Last Statement URL": last_statement_link
            })
            
            if i % 50 == 0:
                print(f"Processed {i} rows...")
                
        except Exception as e:
            print(f"Error parsing row {i}: {e}")
            continue

    # Create data folder if it doesn't exist
    data_folder = "data"
    os.makedirs(data_folder, exist_ok=True)

    # Save to CSV
    filename = os.path.join(data_folder, "texas_execution_dates_with_statements.csv")
    keys = ["Execution Number", "First Name", "Last Name", "TDCJ Number", "Execution Date", "Last Statement", "Last Statement URL"]

    try:
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(data_to_save)
        print(f"Successfully saved {len(data_to_save)} records to {filename}")
        
        # Print a few sample records to verify
        print("\nSample records:")
        for i, record in enumerate(data_to_save[:5]):
            print(f"  {record}")
            
    except Exception as e:
        print(f"Error saving to CSV: {e}")


if __name__ == "__main__":
    main()
