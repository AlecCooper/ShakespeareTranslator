from lang_model import Encoder, Decoder
import tensorflow as tf
import os
from clean import clean_sentence
import pickle

def eval(sentence):

    pass

def translate(sentence):

    # Params
    checkpoint_dir="/check_dir"
    vocab_len = 24193
    embed_dim = 256
    encode_units = 1024
    batch_size = 1
    max_length = 42
    max_seq_length = 15

    # load in the vocab
    with open('rev_vocab.pickle', 'rb') as handle:
        vocab = pickle.load(handle)

    # Create and Encoder and Decoder
    encoder = Encoder(vocab_len, embed_dim, encode_units, batch_size)
    decoder = Decoder(vocab_len, embed_dim, encode_units, batch_size)

    # Create checkpoint
    checkpoint = tf.train.Checkpoint(encoder=encoder, decoder=decoder)

    # Load the last checkpoint
    cwd = os.getcwd()
    checkpoint_dir = cwd + checkpoint_dir
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    # Clean the sentence
    sentence = clean_sentence(sentence, max_length)
    
    # Convert to tensor
    inputs = tf.convert_to_tensor([sentence])

    # Intitalize encoder hidden state, run encoder on  inputs
    encoder_hidden = [tf.zeros((1,encode_units))]
    encoder_out, encoder_hidden = encoder(inputs, encoder_hidden)

    # Decoder hidden state is last encoder hidden state
    decoder_hidden = encoder_hidden

    current_word = tf.expand_dims(tf.convert_to_tensor([inputs[0][0]]),axis=0)

    # The translation result
    result = ""

    # Feed the decoder our sentence
    for word_index in range(0,max_seq_length):
        current_word = tf.expand_dims(tf.convert_to_tensor([inputs[0][word_index]]),axis=0)
        logits, decoder_hidden, attention_weights = decoder(current_word, decoder_hidden, encoder_out)
        
        predicted_id = tf.argmax(logits[0]).numpy()

        # Add space between words
        result += vocab[predicted_id] + " "

        dec_input = tf.expand_dims([predicted_id], 0)

        
    print(result)

translate("This is a test")
