from lang_model import Encoder, Decoder
import tensorflow as tf
import os
from clean import clean_sentence
import pickle
from keras.models import load_model

def translate(sentence):

    # Params
    max_length = 42
    max_seq_length = 15
    encode_units = 1024

    # load in the vocab
    with open('rev_vocab.pickle', 'rb') as handle:
        rev_vocab = pickle.load(handle)

    with open('vocab.pickle', 'rb') as handle:
        vocab = pickle.load(handle)

    # Load the saved models in
    encoder = tf.saved_model.load("encoder")
    decoder = tf.saved_model.load("decoder")

    # Clean the sentence
    sentence = clean_sentence(sentence, max_length)
    
    # Convert to tensor
    inputs = tf.convert_to_tensor([sentence])

    # Load the saved models in
    encoder = tf.saved_model.load("data/model/encoder_5")
    decoder = tf.saved_model.load("data/model/decoder_5")

    # Intitalize encoder hidden state, run encoder on  inputs
    encoder_hidden = [tf.zeros((1, encode_units))]
    encoder_out, encoder_hidden = encoder([inputs, encoder_hidden])

    # Decoder hidden state is last encoder hidden state
    decoder_hidden = encoder_hidden

    # Feed the start token as the first input
    current_word = tf.expand_dims(tf.convert_to_tensor([vocab["<start>"]]),axis=0)

    # The translation result
    result = ""

    # Feed the decoder our sentence
    for word_index in range(0,max_seq_length):

        # Pass input through the decoder
        logits, decoder_hidden, attention_weights = decoder([current_word, decoder_hidden, encoder_out])

        predicted_id = tf.argmax(logits[0]).numpy()

        # Add space between words
        result += rev_vocab[predicted_id] + " "

        # Feed the next decoder state
        dec_input = tf.expand_dims([predicted_id], 0)

    return result

print(translate("This is a test"))
