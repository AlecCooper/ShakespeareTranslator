import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize

# Given a text line, cleans the text for processing
def clean(text):

    # Change text to lowercase
    text = text.lower()

    # Remove text between [],(),{} as it is often does not appear in both original/translation
    start_markers = ["(","[","{"]
    end_markers = [")","]","}"]
    # loop through each possible marker
    for start_marker, end_marker in zip(start_markers, end_markers):
        start = text.find(start_marker)
        end = text.find(end_marker)
        if start != -1 and end != -1:
            text = text[:start] + text[end+1:]

    # Replace - with spaces
    text = text.replace("-", " ")
    text = text.replace("—", " ")

    # Remove quotations
    text = text.replace("\"", "")
    text = text.replace("“","")

    # Change multi spaces to single spaces
    while "  " in text:
        text = text.replace("  ", " ")

    # Remove any leading or trailing whitespace
    text = text.rstrip().lstrip()

    # Check if string is ascii, returning nothing if not
    if not all(ord(c) < 128 for c in text):
        return ""

    return text

# Removes empty or invalid lines from the corpus
def filter_text(corpus):

    # This list contains the valid corpus
    new_corpus = []

    # Loop through every line so we can filter
    for row in corpus:

        # Should we add the line?
        valid = True

        # We filter out empty lines
        if row[2] == "" or row[3] == "":
            valid = False

        # An @ symbol in the line signifies invalidity
        if "@" in row[2] or "@" in row[3]:
            valid = False

        # If valid line, we include it
        if valid:
            new_corpus.append([row[2],row[3]])

    # Turn into numpy array
    new_corpus = np.array(new_corpus)

    return new_corpus

# Tokenize each sentence in the corpus
def tokenize(corpus):

    # list of tokenized lines
    tokenized = []

    # loop through each translation couplet
    for row in corpus:

        # tokenize translation couplet and add
        tokenized.append([word_tokenize(row[0]), word_tokenize(row[1])])

    return tokenized


def main():

    # Read in the corpus file
    corpus = pd.read_csv("corpus.csv")
    corpus = corpus.to_numpy()

    # loop counter
    row_num = 0

    # loop through corpus and clean all the text
    for row in corpus:

        # Clean the text
        corpus[row_num][2] = clean(row[2])
        corpus[row_num][3] = clean(row[3])

        # iterate counter
        row_num += 1

    # Filter out invalid text
    corpus = filter_text(corpus)

    # Tokenize the text
    corpus = tokenize(corpus)

    print(corpus)


main()