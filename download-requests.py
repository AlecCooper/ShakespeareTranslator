from bs4 import BeautifulSoup
import os

# Given the location of a document, returns a 2d list of original and modern lines
def parse_document(document):

    # open document
    with open(document) as fp:
        soup = BeautifulSoup(fp)
    
    # Filter to just the rows comparing translation/original
    paragraphs = soup.findAll("div",class_="comparison-row")

    # Filter out the title at the start of the comparison rows
    paragraphs = paragraphs[1:6] #6 IS FOR TESTING

    # list of oirginal lines and translated lines
    original = []
    translation = []
    
    # Loop through translations
    for paragraph in paragraphs:

        # filter out original lines
        lines = paragraph.findAll("span", class_ = "shakespeare-translation-line")

        # loop through original lines 
        for line in lines:
            original.append(line.string)

        # filter out the translated lines
        lines = paragraph.findAll("p", class_ = "speaker-text")

        # loop through translated lines 
        for line in lines:
            if line.string != None:
                translation.append(line.string)

    return original, translation
    
# Lists holding the original, translated lines
original = []
translation = []

# Directory containing the webpages
rootdir = os.getcwd() + "/www.litcharts.com/shakescleare/shakespeare-translations"

# We loop through all the directories and parse all the files
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        print("Parsing" + os.path.join(subdir, file))
        new_original, new_translation = parse_document(os.path.join(subdir, file))

        print(new_original)

        #original = original.extend(new_original)
        #translation = translation.extend(new_translation)
        
# Save to text documents
with open("original.txt", "w") as f:
    for line in original:
        f.write("%s\n" % line)

with open("translation.txt", "w") as f:
    for line in translation:
        f.write("%s\n" % line)



        


