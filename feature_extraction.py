from bs4 import BeautifulSoup
import os
import features as fe

# DEFINE A FUNCTION THAT CREATES A VECTOR BY RUNNING ALL FEATURE FUNCTIONS
def create_vector(soup, url):
    # 1. Extract 48 Content-Based Features (already in your features.py)
    content_vector = [
        fe.has_title(soup),
        fe.has_input(soup),
        fe.has_button(soup),
        fe.has_image(soup),
        fe.has_submit(soup),
        fe.has_link(soup),
        fe.has_password(soup),
        fe.has_email_input(soup),
        fe.has_hidden_element(soup),
        fe.has_audio(soup),
        fe.has_video(soup),
        fe.number_of_inputs(soup),
        fe.number_of_buttons(soup),
        fe.number_of_images(soup),
        fe.number_of_option(soup),
        fe.number_of_list(soup),
        fe.number_of_TH(soup),
        fe.number_of_TR(soup),
        fe.number_of_href(soup),
        fe.number_of_paragraph(soup),
        fe.number_of_script(soup),
        fe.length_of_title(soup),
        fe.has_h1(soup),
        fe.has_h2(soup),
        fe.has_h3(soup),
        fe.length_of_text(soup),
        fe.number_of_clickable_button(soup),
        fe.number_of_a(soup),
        fe.number_of_img(soup),
        fe.number_of_div(soup),
        fe.number_of_figure(soup),
        fe.has_footer(soup),
        fe.has_form(soup),
        fe.has_text_area(soup),
        fe.has_iframe(soup),
        fe.has_text_input(soup),
        fe.number_of_meta(soup),
        fe.has_nav(soup),
        fe.has_object(soup),
        fe.has_picture(soup),
        fe.number_of_sources(soup),
        fe.number_of_span(soup),
        fe.number_of_table(soup),
        fe.check_form_action(soup, url),                     # Binary: 1 if suspicious
        fe.ratio_null_hyperlinks(soup),                     # Float: 0.0 to 1.0
        fe.get_external_ratio(soup, url, 'img', 'src'),     # Float: Ratio of external images
        fe.get_external_ratio(soup, url, 'link', 'href'),    # Float: Ratio of external CSS
        fe.get_external_ratio(soup, url, 'script', 'src')    # Float: Ratio of external JS
    ]
    
    # 2. Extract 13 URL-Based Lexical Features
    url_vector = fe.extract_url_features(url)
    
    # 3. Return Combined Vector (Total: 61 Features)
    return content_vector + url_vector