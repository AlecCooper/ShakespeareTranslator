from lang_model import Encoder, Decoder
import tensorflow as tf
import os

def eval(sentence):

    pass


def translate(sentence):

    # Params
    checkpoint_dir="/check_dir"
    vocab_len = 24191
    embed_dim = 256
    encode_units = 128
    batch_size = 10

    # Create and Encoder and Decoder
    encoder = Encoder(vocab_len, embed_dim, encode_units, batch_size)
    decoder = Decoder(vocab_len, embed_dim, encode_units, batch_size)

    # Create checkpoint
    checkpoint = tf.train.Checkpoint(encoder=encoder, decoder=decoder)

    # Load the last checkpoint
    cwd = os.getcwd()
    checkpoint_dir = cwd + checkpoint_dir
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    
    print("test")

translate("test")
