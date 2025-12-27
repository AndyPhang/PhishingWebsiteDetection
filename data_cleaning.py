import pandas as pd
from urllib.parse import urlparse, urlunparse

def normalize_url(url):
    """Standardizes URLs for effective deduplication."""
    if not isinstance(url, str):
        return url
    
    # 1. Lowercase the entire URL
    url = url.lower().strip()
    
    # 2. Parse URL to components
    parsed = urlparse(url)
    
    # 3. Remove trailing slashes from the path
    path = parsed.path.rstrip('/')
    
    # 4. Reconstruct the URL without query/fragment (optional but recommended for deduplication)
    # Keeping only scheme, netloc, and cleaned path
    normalized = urlunparse((parsed.scheme, parsed.netloc, path, '', '', ''))
    
    return normalized

def clean_dataset(input_file, output_file):
    print(f"--- Cleaning: {input_file} ---")
    df = pd.read_csv(input_file)
    initial_count = len(df)
    
    # Apply normalization
    df['normalized_url'] = df['url'].apply(normalize_url)
    
    # Remove duplicates based on the normalized version
    # 'keep=first' ensures we retain the first instance
    df = df.drop_duplicates(subset=['normalized_url'], keep='first')
    
    final_count = len(df)
    print(f"Removed {initial_count - final_count} duplicates.")
    
    # Drop the helper column and save
    df = df.drop(columns=['normalized_url'])
    df.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}\n")

# Run cleaning on your datasets
clean_dataset("Dataset/legitimate_list.csv", "Dataset/cleaned_legitimate_list.csv")
clean_dataset("Dataset/phishing_list.csv", "Dataset/cleaned_phishing_list.csv")