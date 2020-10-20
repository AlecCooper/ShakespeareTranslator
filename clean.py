import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
import tensorflow as tf

## PARAMS
max_length = 16

# Given a text line, cleans the text for processing
def clean(text):

  text = str(text)

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
    
  # Add start and end tokens
  text = "<start> " + text + " <end>"

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

# Removes empty or invalid lines from the corpus
def filter_text(corpus):

    # This list contains the valid corpus
    new_corpus = []

    # Loop through every line so we can filter
    for row in corpus:

        # Should we add the line?
        valid = True

        # Make sure not nonetype
        if row[2] is None or row[3] is None:
          valid = False
          break

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

# Creates a dataset to feed into the language model
def create_dataset():

    # Read in the corpus file
    corpus = pd.read_csv("data/corpus.csv")
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

    # Seperate out input and target languages
    target_lang = corpus[:,0]
    input_lang = corpus[:,1]

    # Create tokenizers and fit on text
    target_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    target_tokenizer.fit_on_texts(target_lang)
    
    input_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    input_tokenizer.fit_on_texts(input_lang)

    # Tokenize our corupus
    target_tensor = target_tokenizer.texts_to_sequences(target_lang)
    input_tensor = input_tokenizer.texts_to_sequences(input_lang)

    filtered_input = []
    filtered_target = []

    # Remove lines larger than max length
    for i in range(len(input_tensor)):
      if len(input_tensor[i]) <= max_length and len(target_tensor[i]) <= max_length:
        filtered_input.append(input_tensor[i])
        filtered_target.append(target_tensor[i])


    # Pad the tensors
    target_tensor = tf.keras.preprocessing.sequence.pad_sequences(filtered_target, padding='post')
    input_tensor = tf.keras.preprocessing.sequence.pad_sequences(filtered_input, padding='post')

    input_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.int32)
    target_tensor = tf.convert_to_tensor(target_tensor, dtype=tf.int32)

    return [input_tensor, target_tensor], [input_tokenizer, target_tokenizer]
