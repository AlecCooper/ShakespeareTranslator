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
def tokenize(corpus, max_length):

    # list of tokenized lines
    tokenized = []

    # loop through each translation couplet
    for row in corpus:

        # tokenize translation couplet
        tokenized_lines = (word_tokenize(row[0]), word_tokenize(row[1]))

        # Make sure they are under the max allowed length
        if len(tokenized_lines[0]) <= max_length and len(tokenized_lines[1]) <= max_length:
            tokenized.append([tokenized_lines[0],tokenized_lines[1]])

    return tokenized

# Given a list of tokenized words, creates a word embedding
# aka a unique integer is assinged to every unique word in the vocab
def word_embed(lines):

    # This dictonary defines our vocab
    vocab = {}

    # The integer we map each new token to
    map_int = 0

    # the list of embedded lines 
    embedded = []

    # loop through every line and embedd
    for line in lines:

        # the integerized tokens are stored in this list
        new_line = []

        for token in line:
            
            # If the token is already included in the vocab,
            # we can map it to an integer
            if token in vocab:
                new_line.append(vocab[token])

            # If the token is not in the vocab, we map it to a
            # new integer
            else:
                # Create new mapping
                vocab[token] = map_int

                new_line.append(map_int) 
                map_int += 1


        embedded.append(new_line)


    return embedded

def main():

    # Hyperparamaters
    max_length = 40

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
    corpus = tokenize(corpus, max_length)

    # Embedd the text
    original = word_embed(corpus[0])
    translation = word_embed(corpus[1])

    print(original)

    

main()

