import pandas as pd
import re
from datetime import datetime


def parse_email_content(email_content):
    """
    Parse email content into structured fields.
    Returns: dict with Sender_Email, Receiver_Email, Date, Day, Body, Links
    """
    if pd.isna(email_content):
        return {
            'Sender_Email': None,
            'Receiver_Email': None,
            'Date': None,
            'Day': None,
            'Body': None,
            'Links': None
        }

    # Convert to string and handle case-insensitive matching
    content = str(email_content)

    # Extract From address (case-insensitive)
    from_match = re.search(r'From:\s*(.+?)(?:\n|$)', content, re.IGNORECASE)
    from_address = from_match.group(1).strip() if from_match else None

    # Extract To address (case-insensitive)
    to_match = re.search(r'To:\s*(.+?)(?:\n|$)', content, re.IGNORECASE)
    to_address = to_match.group(1).strip() if to_match else None

    # Extract Date (case-insensitive)
    date_match = re.search(r'Date:\s*(.+?)(?:\n|$)', content, re.IGNORECASE)
    date_str = date_match.group(1).strip() if date_match else None

    # Parse date and extract day of week
    day_of_week = None
    if date_str:
        try:
            # Try parsing as M/D/YYYY format
            parsed_date = datetime.strptime(date_str, '%m/%d/%Y')
            day_of_week = parsed_date.strftime('%A')
        except:
            try:
                # Try other common formats
                parsed_date = datetime.strptime(date_str, '%d/%m/%Y')
                day_of_week = parsed_date.strftime('%A')
            except:
                pass

    # Extract subject (first line after Date that's not empty)
    subject = None
    if date_match:
        # Get content after the date line
        after_date = content[date_match.end():].strip()
        lines = after_date.split('\n')
        for line in lines:
            line = line.strip()
            if line:
                subject = line
                break

    # Extract body (everything after subject)
    body = None
    if subject and date_match:
        # Find where subject appears after the date
        after_date = content[date_match.end():].strip()
        subject_pos = after_date.find(subject)
        if subject_pos != -1:
            # Body is everything after the subject line
            body = after_date[subject_pos + len(subject):].strip()

    # Extract links from the entire content
    links = []
    # Pattern for URLs in plain text (http:// or https://)
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    links.extend(re.findall(url_pattern, content, re.IGNORECASE))

    # Pattern for URLs in HTML href attributes
    href_pattern = r"href=['\"]([^'\"]+)['\"]"
    links.extend(re.findall(href_pattern, content, re.IGNORECASE))

    # Remove duplicates and join with semicolon
    links = list(dict.fromkeys(links))  # Preserve order while removing duplicates
    links_str = '; '.join(links) if links else None

    return {
        'Sender_Email': from_address,
        'Receiver_Email': to_address,
        'Date': date_str,
        'Day': day_of_week,
        'Body': body,
        'Links': links_str
    }


# Read the CSV file
df = pd.read_csv('synthetic_phishing_dataset_100k.csv')

print(f"Original dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Parse each email content
parsed_data = df['Email_Content'].apply(parse_email_content)

# Convert the parsed data to a DataFrame
parsed_df = pd.DataFrame(parsed_data.tolist())

print("\nParsed columns created:")
print(parsed_df.columns.tolist())

# Combine with original data
# Drop the original Email_Content column and add the new parsed columns
result_df = df.drop('Email_Content', axis=1)
result_df = pd.concat([result_df, parsed_df], axis=1)

# Reorder columns for better readability
cols = ['Email_Subject', 'Sender_Email', 'Receiver_Email', 'Date', 'Day', 'Body', 'Links']
# Add any other columns that might exist
other_cols = [col for col in result_df.columns if col not in cols]
result_df = result_df[cols + other_cols]

# Save to new CSV
output_filename = 'parsed_phishing_dataset.csv'
result_df.to_csv(output_filename, index=False)

print(f"\nParsed dataset shape: {result_df.shape}")
print(f"\nFinal columns: {result_df.columns.tolist()}")
print(f"\nSaved to: {output_filename}")

# Display sample of parsed data
print("\n" + "=" * 80)
print("Sample of parsed data:")
print("=" * 80)
pd.set_option('display.max_colwidth', 50)
print(result_df[['Email_Subject', 'Sender_Email', 'Date', 'Day', 'Links']].head(10))

# Show some statistics
print("\n" + "=" * 80)
print("Parsing Statistics:")
print("=" * 80)
print(f"Successfully parsed From addresses: {result_df['Sender_Email'].notna().sum()}")
print(f"Successfully parsed To addresses: {result_df['Receiver_Email'].notna().sum()}")
print(f"Successfully parsed Dates: {result_df['Date'].notna().sum()}")
print(f"Successfully extracted Days: {result_df['Day'].notna().sum()}")
print(f"Successfully extracted Bodies: {result_df['Body'].notna().sum()}")
print(f"Emails with links: {result_df['Links'].notna().sum()}")

# Show a few examples of extracted links
print("\n" + "=" * 80)
print("Sample of extracted links:")
print("=" * 80)
links_sample = result_df[result_df['Links'].notna()][['Email_Subject', 'Links']].head(5)
for idx, row in links_sample.iterrows():
    print(f"\nSubject: {row['Email_Subject']}")
    print(f"Links: {row['Links']}")