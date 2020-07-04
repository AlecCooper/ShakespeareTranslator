from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
import os

def init_driver():
    # Setup headless chrome
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--window-size=1024x1400")
    chrome_driver = os.path.join(os.getcwd(), "chromedriver")
    driver = webdriver.Chrome(options=chrome_options, executable_path=chrome_driver)
    return driver

# Retrieves the url and returns it as a soup
def make_soup(url, driver):

    # Download the url
    print("Downloading " + url)
    driver.get(url)

    # Parse with beautiful soup
    print("Parsing " + url)
    source = driver.page_source
    soup = BeautifulSoup(source,"lxml")

    return soup

# This method extrats a string from between html tags
def extract_string(tags):

    if tags.string != None:
        return tags.string
    else:
        # We must remove names between <i> tags
        if tags.find("i") != None:

            # extract the text with no italizited text
            text = tags.text

            # Remove text between [], as it is often does not appear in both original/translation
            start = text.find("[")
            end = text.find("]")
            if start != -1 and end != -1:
                text = text[:start] + text[end + 1:]

            # remove any right spaces and return
            return text.rstrip(" ")
                
        # There could be some sort of annotation on the line, in which case we mark it for later removal
        # (Maybe fix later?)
        return "@" 

# Given a list of lines, extracts and concatenates the lines by data id
def extract_lines(lines):

    # This dictonary holds the extracted lines, with the key being the line's data id
    line_dict = {}

    # If there is more than one line, we need to loop through and extract them
    data_id = lines[0]["data-id"]

    # temp var to hold the strings we are concatenating
    line_string = extract_string(lines[0])

    if len(lines) > 1:
        # Loop through the remaining lines, concatenating identical ids
        for line in lines[1:]:

            # concate if same data id
            if line["data-id"] == data_id:
                               
                line_string += extract_string(line)      
            else:
                # Add the fully concatenated line
                line_dict[data_id] = line_string

                # reset our holding variables
                data_id = line["data-id"]       
                line_string = extract_string(line)

            # Make sure we've added the last line
        if not data_id in line_dict:
            line_dict[data_id] = line_string

    else:       
        line_dict[data_id] = line_string

    return line_dict

# Given the location of a document, returns a 2d list of original and modern lines
def parse_document(soup):
    
    # Filter to just the rows comparing translation/original
    rows = soup.findAll("div",class_="comparison-row")

    # Filter out the title at the start of the comparison rows
    rows = rows[1:]

    # dict of oirginal lines and translated lines
    original = {}
    translation = {}

    # Loop through translations
    for row in rows:

        # Seperate the original and translated sections
        original_content_col = row.div
        translated_content_col = row.div.next_sibling.next_sibling

        # Find the speaker text
        original_speaker_text = original_content_col.find("p", class_="speaker-text")
        translated_speaker_text = translated_content_col.find("p", class_="speaker-text")
        
        # Determine if both versions have a speaker text, if so we extract it
        if original_speaker_text != None and translated_speaker_text != None:
            
            # Extract the lines
            original_lines = original_speaker_text.findAll("span", class_=["line-mapping mapped","line-mapping"])
            translated_lines = translated_speaker_text.findAll("span",class_=["line-mapping mapped","line-mapping"])

            # We must concatenate all the lines with the same data id together
            # to get the full line
            if len(original_lines) > 0 and len(translated_lines) > 0:
                
                # Extract the lines from the now, update the dictonary with the new entries
                original.update(extract_lines(original_lines))
                translation.update(extract_lines(translated_lines))

    # A list of translated pairs
    corpus = []

    # Loop through translation, orignal and match data id
    for data_id in original:
        
        # Is there a matching id in the other dict?
        if data_id in translation:
            corpus.append((data_id, original[data_id], translation[data_id]))

    return corpus

# Saves the corpus as a csv file
def save(corpus, name):

    # Create dataframe
    corpus_df = pd.DataFrame(corpus, columns=["id", "original", "translation"])

    # Save dataframe
    corpus_df.to_csv(name)
    
driver = init_driver()
soup = make_soup("https://www.litcharts.com/shakescleare/shakespeare-translations/coriolanus/act-1-scene-1",driver)
test1 = parse_document(soup)
save(test1,"test.csv")





        


