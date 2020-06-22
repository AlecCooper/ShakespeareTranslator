from bs4 import BeautifulSoup

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
        #print(lines)
        # loop through translated lines 
        for line in lines:
            if line.string != None:
                translation.append(line.string)


    return original, translation
    
