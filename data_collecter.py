import pandas as pd
import requests as re
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import feature_extraction as fe
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib3.exceptions import InsecureRequestWarning
from urllib3 import disable_warnings
import warnings
import os

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
disable_warnings(InsecureRequestWarning)

# --- CONFIGURATION ---
MAX_WORKERS = 30  
TIMEOUT = 7       

# Updated COLUMNS to include the 61 features + URL
COLUMNS = [
    'has_title', 'has_input', 'has_button', 'has_image', 'has_submit', 'has_link',
    'has_password', 'has_email_input', 'has_hidden_element', 'has_audio', 'has_video',
    'number_of_inputs', 'number_of_buttons', 'number_of_images', 'number_of_option',
    'number_of_list', 'number_of_th', 'number_of_tr', 'number_of_href', 'number_of_paragraph',
    'number_of_script', 'length_of_title', 'has_h1', 'has_h2', 'has_h3', 'length_of_text',
    'number_of_clickable_button', 'number_of_a', 'number_of_img', 'number_of_div',
    'number_of_figure', 'has_footer', 'has_form', 'has_text_area', 'has_iframe',
    'has_text_input', 'number_of_meta', 'has_nav', 'has_object', 'has_picture',
    'number_of_sources', 'number_of_span', 'number_of_table',
    # Advanced Content Features
    'form_action_suspicious', 'null_hyperlinks_ratio', 'external_img_ratio', 
    'external_css_ratio', 'external_js_ratio',
    # URL Lexical Features
    'url_len', 'host_len', 'dot_count', 'hyphen_count', 'is_ip', 'has_at',
    'double_slash', 'dir_count', 'http_in_host', 'has_keyword', 'digit_count',
    'is_shortened', 'risky_tld', 
    'URL' # Final column for identification
]

def process_single_url(url):
    """Processes a single URL by fetching content and extracting the full feature vector."""
    try:
        # Added User-Agent to improve scraping success rate
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = re.get(url, verify=False, timeout=TIMEOUT, headers=headers)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            # Pass BOTH soup and url as required by updated feature_extraction.py
            vector = fe.create_vector(soup, url)
            vector.append(str(url))
            return vector
    except Exception:
        pass
    return None

def scrape_and_save(input_path, output_path, label, limit, fix_protocol=False):
    """Workflow to scrape a specific limit of unique, cleaned URLs."""
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found. Ensure data_cleaning.py was run.")
        return

    print(f"\n--- Scraping {limit} entries from: {input_path} (Label: {label}) ---")
    data_frame = pd.read_csv(input_path)
    # Apply your research limit (60k for Legitimate, 40k for Phishing)
    data_frame = data_frame.head(limit)
    
    urls_to_process = data_frame['url'].tolist()

    if fix_protocol:
        collection_list = [
            "https://" + str(url) if not str(url).startswith(('http://', 'https://')) else str(url)
            for url in urls_to_process
        ]
    else:
        collection_list = [str(url) for url in urls_to_process]

    data_list = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single_url, url): url for url in collection_list}

        # Progress bar to track live scraping completion
        for future in tqdm(as_completed(futures), total=len(collection_list), desc="Scraping Progress", unit="url"):
            result = future.result()
            if result:
                data_list.append(result)

    # Convert to DataFrame and add the target label
    df_result = pd.DataFrame(data=data_list, columns=COLUMNS)
    df_result['label'] = label

    # Save to Dataset/ folder
    df_result.to_csv(output_path, index=False)
    print(f"Successfully saved {len(df_result)} records to {output_path}")

# --- EXECUTION ---
# 1. Process 60,000 Legitimate URLs
scrape_and_save(
    input_path="Dataset/cleaned_legitimate_list.csv",
    output_path="Dataset/structured_legitimate_list.csv",
    label=0,
    limit=60000,
    fix_protocol=True
)

# 2. Process 40,000 Phishing URLs
scrape_and_save(
    input_path="Dataset/cleaned_phishing_list.csv",
    output_path="Dataset/structured_phishing_list.csv",
    label=1,
    limit=40000,
    fix_protocol=False
)