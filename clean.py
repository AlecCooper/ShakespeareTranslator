import json, argparse
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import pickle

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
    original = []
    translated = []

    # loop through each translation couplet
    for row in corpus:

        # tokenize translation couplet
        tokenized_lines = (word_tokenize(row[0]), word_tokenize(row[1]))

        # Insert start and end tokens
        tokenized_lines[0].append("<end>")
        tokenized_lines[0].insert(0, "<start>")
        tokenized_lines[1].append("<end>")
        tokenized_lines[1].insert(0, "<start>")

        # Make sure they are under the max allowed length
        if len(tokenized_lines[0]) <= max_length and len(tokenized_lines[1]) <= max_length:
            original.append(tokenized_lines[0])
            translated.append(tokenized_lines[1])

    return original, translated

# Given a list of tokenized words and a dictonary with a word mapping
# a unique integer is assinged to every unique word in the vocab
def word_embed(lines, vocab, rev_vocab, map_int):

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
                rev_vocab[map_int] = token

                new_line.append(map_int) 
                map_int += 1

        embedded.append(new_line)

    return embedded, vocab, rev_vocab, map_int

def clean_sentence(text, max_length):

    # load in the vocab
    with open('vocab.pickle', 'rb') as handle:
        vocab = pickle.load(handle)

    # clean the text
    text = clean(text)

    # Use nltk to tokenize the sentence
    text = word_tokenize(text)

    # list we will append the token indices to
    embedded_text = []

    # embedd each token
    for token in text:
        embedded_text.append(vocab[token])

    # Pad the sentence
    while len(embedded_text) < max_length:
        embedded_text.append(0)

    # Turn into numpy array
    sentence = np.array(embedded_text)

    return sentence

def main():

    # Extract max sentence length from the hyperparamater file
    max_length = 42

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
    original, translation = tokenize(corpus, max_length)

    # Embedd the text
    vocab = {}    # The dictonary from which we map tokens to ints  
    rev_vocab = {}
    map_int = 1    # The integer we start mapping to
    original, vocab, rev_vocab, map_int = word_embed(original, vocab, rev_vocab, map_int)
    translation, vocab, rev_vocab, map_int = word_embed(translation, vocab, rev_vocab, map_int)

    # Pad the text
    original = tf.keras.preprocessing.sequence.pad_sequences(original, padding="post")
    translation = tf.keras.preprocessing.sequence.pad_sequences(translation, padding="post")

    # Save data
    np.save("data/original", original)
    np.save("data/translation", translation)

    # Save the dictonary 
    with open('vocab.pickle', 'wb') as handle:
        pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open('rev_vocab.pickle', 'wb') as handle:
        pickle.dump(rev_vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Print our vocab size
    print("Vocab size: " + str(len(vocab)))

main()

