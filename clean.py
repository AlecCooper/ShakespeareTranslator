import pandas as pd

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

    return text

def main():

    # Read in the corpus file
    corpus = pd.read_csv("corpus.csv")

    # loop through corpus and clean all the text
    for row in corpus:

        pass

print(clean("What worser place can I beg in your love—"))