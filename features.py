import re
from urllib.parse import urlparse

# --- EXISTING CONTENT-BASED FEATURES ---
def has_title(soup):
    if soup.title is None: return 0
    return 1 if len(soup.title.text) > 0 else 0

def has_input(soup):
    return 1 if len(soup.find_all("input")) > 0 else 0

def has_button(soup):
    return 1 if len(soup.find_all("button")) > 0 else 0

def has_image(soup):
    return 1 if len(soup.find_all("image")) > 0 else 0

def has_submit(soup):
    for button in soup.find_all("input"):
        if button.get("type") == "submit": return 1
    return 0

def has_link(soup):
    return 1 if len(soup.find_all("link")) > 0 else 0

def has_password(soup):
    for input_tag in soup.find_all("input"):
        if (input_tag.get("type") or input_tag.get("name") or input_tag.get("id")) == "password": return 1
    return 0

def has_email_input(soup):
    for input_tag in soup.find_all("input"):
        if (input_tag.get("type") or input_tag.get("id") or input_tag.get("name")) == "email": return 1
    return 0

def has_hidden_element(soup):
    for input_tag in soup.find_all("input"):
        if input_tag.get("type") == "hidden": return 1
    return 0

def has_audio(soup):
    return 1 if len(soup.find_all("audio")) > 0 else 0

def has_video(soup):
    return 1 if len(soup.find_all("video")) > 0 else 0

def number_of_inputs(soup):
    return len(soup.find_all("input"))

def number_of_buttons(soup):
    return len(soup.find_all("button"))

def number_of_images(soup):
    return len(soup.find_all("image")) + len([m for m in soup.find_all("meta") if m.get("name") == "image"])

def number_of_option(soup):
    return len(soup.find_all("option"))

def number_of_list(soup):
    return len(soup.find_all("li"))

def number_of_TH(soup):
    return len(soup.find_all("th"))

def number_of_TR(soup):
    return len(soup.find_all("tr"))

def number_of_href(soup):
    return len([link for link in soup.find_all("link") if link.get("href")])

def number_of_paragraph(soup):
    return len(soup.find_all("p"))

def number_of_script(soup):
    return len(soup.find_all("script"))

def length_of_title(soup):
    return len(soup.title.text) if soup.title else 0

def has_h1(soup):
    return 1 if len(soup.find_all("h1")) > 0 else 0

def has_h2(soup):
    return 1 if len(soup.find_all("h2")) > 0 else 0

def has_h3(soup):
    return 1 if len(soup.find_all("h3")) > 0 else 0

def length_of_text(soup):
    return len(soup.get_text())

def number_of_clickable_button(soup):
    return len([b for b in soup.find_all("button") if b.get("type") == "button"])

def number_of_a(soup):
    return len(soup.find_all("a"))

def number_of_img(soup):
    return len(soup.find_all("img"))

def number_of_div(soup):
    return len(soup.find_all("div"))

def number_of_figure(soup):
    return len(soup.find_all("figure"))

def has_footer(soup):
    return 1 if len(soup.find_all("footer")) > 0 else 0

def has_form(soup):
    return 1 if len(soup.find_all("form")) > 0 else 0

def has_text_area(soup):
    return 1 if len(soup.find_all("textarea")) > 0 else 0

def has_iframe(soup):
    return 1 if len(soup.find_all("iframe")) > 0 else 0

def has_text_input(soup):
    for input_tag in soup.find_all("input"):
        if input_tag.get("type") == "text": return 1
    return 0

def number_of_meta(soup):
    return len(soup.find_all("meta"))

def has_nav(soup):
    return 1 if len(soup.find_all("nav")) > 0 else 0

def has_object(soup):
    return 1 if len(soup.find_all("object")) > 0 else 0

def has_picture(soup):
    return 1 if len(soup.find_all("picture")) > 0 else 0

def number_of_sources(soup):
    return len(soup.find_all("source"))

def number_of_span(soup):
    return len(soup.find_all("span"))

def number_of_table(soup):
    return len(soup.find_all("table"))

# --- DOMAIN-AWARE CONTENT FEATURES ---
def get_external_ratio(soup, base_url, tag, attr):
    items = soup.find_all(tag)
    if not items: return 0.0
    base_domain = urlparse(base_url).netloc
    external = sum(1 for i in items if i.get(attr) and urlparse(i.get(attr)).netloc and urlparse(i.get(attr)).netloc != base_domain)
    return external / len(items)

def check_form_action(soup, base_url):
    base_domain = urlparse(base_url).netloc
    for form in soup.find_all('form'):
        action = form.get('action', '').strip()
        if not action or action in ["#", "about:blank"]: return 1
        if urlparse(action).netloc and urlparse(action).netloc != base_domain: return 1
    return 0

def ratio_null_hyperlinks(soup):
    links = soup.find_all('a')
    if not links: return 0.0
    null_links = sum(1 for l in links if l.get('href', '').strip() in ['#', 'javascript:void(0)', 'javascript:void(0);', ''])
    return null_links / len(links)

# --- NEW URL-BASED LEXICAL FEATURES ---
def extract_url_features(url):
    parsed = urlparse(url)
    hostname = parsed.netloc
    return [
        len(url),                         # URL Length
        len(hostname),                    # Hostname Length
        hostname.count('.'),              # Dot count in host
        url.count('-'),                   # Hyphen count
        1 if re.search(r'\d+\.\d+\.\d+\.\d+', hostname) else 0, # IP address?
        1 if '@' in url else 0,           # Presence of @
        1 if url.rfind('//') > 7 else 0,  # Redirection //
        parsed.path.count('/'),           # Subdirectory count
        1 if 'http' in hostname else 0,   # HTTP in hostname
        1 if any(kw in url.lower() for kw in ['login', 'verify', 'bank', 'secure']) else 0, # Keywords
        sum(c.isdigit() for c in url),    # Digit count
        1 if re.search(r"bit\.ly|goo\.gl|t\.co|tinyurl", url) else 0, # Shortener
        1 if hostname.split('.')[-1] in ['tk', 'ml', 'ga', 'cf', 'gq'] else 0 # Risky TLD
    ]