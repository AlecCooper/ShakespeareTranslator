import argparse, json
import numpy as np
import tensorflow as tf
from lang_model import Encoder
from lang_model import Decoder

# The loss function we use to calculate training loss
def loss_func(actual, pred, loss_obj):

    # Calculate the cross entropy with our loss object
    loss = loss_obj(actual, pred)

    # Masking on end of sentence marker
    mask = tf.math.logical_not(tf.math.equal(0,actual))
    mask = tf.cast(mask,dtype=loss.dtype)

    # Apply the mask
    loss = loss * mask

    # Reduce to mean across dims
    loss = tf.reduce_mean(loss)

    return loss

def train(num_epochs, lr, batch_size, embed_dim, encode_units, dataset):

    # Extract the embedding_dim from the data set
    vocab_len = 24191     # note: come up with more elegent way to grab this num

    @tf.function
    def train_step(input_seq, target, encoder_hidden, loss_obj):

        with tf.GradientTape() as tape:

            # Initalize our loss to 0
            loss = 0
            
            # Run the encoder on the input
            encoder_output, encoder_hidden = encoder(input_seq, encoder_hidden)

            # Initalize first decoder hidden states to final encoder hidden states
            decoder_hidden = encoder_hidden

            decoder_input = tf.expand_dims(target[:,0],axis=1)

            # Target forcing
            for t in range(1, target.shape[1]):

                # Pass encoder output into decoder
                prediction, decoder_hidden, attention_weights = decoder(decoder_input, decoder_hidden, encoder_output)
                
                # Calculate loss
                loss += loss_func(target[:, t], prediction, loss_obj)
                
                # Teacher forcing
                decoder_input = tf.expand_dims(target[:, t], 1)

            # The total loss of the batch
            batch_loss = (loss / int(target.shape[1]))

            # Train by applying gradients
            variables = encoder.trainable_variables + decoder.trainable_variables
            gradients = tape.gradient(loss, variables)
            optim.apply_gradients(zip(gradients, variables))

        return batch_loss

    # Define the encoder and decoder
    encoder = Encoder(vocab_len, embed_dim, encode_units, batch_size)
    decoder = Decoder(vocab_len, embed_dim, encode_units, batch_size)

    # Define the optimizer
    optim = tf.keras.optimizers.Adam(learning_rate=lr)

    # Object used to calculate loss
    loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,reduction="none")

    # Training loop
    for epoch in range(1,num_epochs+1):

        # Initalize the hidden state of our encoder
        encoder_hidden = encoder.initalize_hidden()

        total_loss = 0
        
        # Loop through our minibatches
        for batch in range(int(len(original)/batch_size)):

            input_batch, target_batch = next(iter(dataset))
            batch_loss = train_step(input_batch, target_batch, encoder_hidden, loss_obj)
            total_loss += batch_loss

            # Verbose output
            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch, batch, batch_loss.numpy()))


if __name__ == "__main__":

    # Get Command Line Arguments
    parser = argparse.ArgumentParser(description="Shakespeare Translator in TensorFlow")
    parser.add_argument("data", metavar="data_dir", help="Folder containing numpy files of original and translated data", type=str)
    parser.add_argument("params",metavar="param_file.json",help="location of hyperparamater json", type=str)
    parser.add_argument('-c', type=str, default="/training_checkpoints", metavar='/training_checkpoints',help='Enable training checkpoints stored to the given dir')
    args = parser.parse_args()

    # Hyperparameters from json file
    with open(args.params) as paramfile:
        hyper = json.load(paramfile)

    # load data
    print("Loading data")
    original = np.load(args.data + "/original.npy")
    translation = np.load(args.data + "/translation.npy")
    print("Done")

    # Number of training epochs for out training loop
    num_epochs = hyper["epochs"]
    # Batch size
    batch_size = hyper["batch size"]
    # Learning rate
    lr = hyper["learning rate"]
    # Embedding dimension
    embed_dim = hyper["embedding dimension"]
    # Numbe of units in our encoder
    encode_units = hyper["encoder units"]

    # Create Tensorflow dataset
    dataset = tf.data.Dataset.from_tensor_slices((original, translation))
    dataset = dataset.batch(batch_size, drop_remainder=True)

    # Train the model
    train(num_epochs, lr, batch_size, embed_dim, encode_units, dataset)

    print(np.shape(translation))
    print(np.shape(original))

    