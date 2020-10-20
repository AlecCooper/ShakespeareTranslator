import argparse, json
import numpy as np
import tensorflow as tf
import os
import time
from lang_model import Encoder
from lang_model import Decoder
from clean import create_dataset, clean_sentence

# The loss function we use to calculate training loss
def loss_func(actual, pred):

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

def train(num_epochs, steps_per_epoch, lr, batch_size, embed_dim, encode_units):

  @tf.function
  def train_step(input_seq, target, encoder_hidden):

    # Initalize our loss to 0
    loss = 0

    with tf.GradientTape() as tape:
            
      # Run the encoder on the input
      encoder_output, encoder_hidden = encoder(input_seq, encoder_hidden)

      # Initalize first decoder hidden states to final encoder hidden states
      decoder_hidden = encoder_hidden

      decoder_input = tf.expand_dims([target_lang.word_index['<start>']] * batch_size, 1)

      # Target forcing
      for t in range(1, target.shape[1]):

        # Pass encoder output into decoder
        prediction, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_output)
                
        # Calculate loss
        loss += loss_func(target[:, t], prediction)
                
        # Teacher forcing
        decoder_input = tf.expand_dims(target[:, t], 1)

      # The total loss of the batch
      batch_loss = (loss / int(target.shape[1]))

      # Train by applying gradients
      variables = encoder.trainable_variables + decoder.trainable_variables
      gradients = tape.gradient(loss, variables)
      optim.apply_gradients(zip(gradients, variables))

      return batch_loss

  # Training loop
  for epoch in range(1, num_epochs+1):

    # Time the epochs
    start = time.time()

    # Initalize the hidden state of our encoder
    encoder_hidden = encoder.initalize_hidden()

    total_loss = 0
        
    # Loop through our minibatches
    for (batch, (input_batch, target_batch)) in enumerate(dataset.take(steps_per_epoch)):

      batch_loss = train_step(input_batch, target_batch, encoder_hidden)
      total_loss += batch_loss

      # Verbose output
      if batch % 100 == 0:
        print('Epoch {} Batch {} Loss {:.4f}'.format(epoch, batch, batch_loss.numpy()))

    # Save checkpoint
    checkpoint.save(file_prefix = checkpoint_prefix)

    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

if __name__ == "__main__":

    # Get Command Line Arguments
    parser = argparse.ArgumentParser(description="Shakespeare Translator in TensorFlow")
    parser.add_argument("data", metavar="/data_dir", help="Folder containing the corpus", type=str)
    parser.add_argument("params",metavar="param_file.json",help="location of hyperparamater json", type=str)
    args = parser.parse_args()

    # Hyperparameters from json file
    with open(args.params) as paramfile:
        hyper = json.load(paramfile)

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
    # Max sentence length (in words)
    max_length = 16

    # Create Tensorflow dataset
    tensors, tokenizers = create_dataset()
    input_tensor, target_tensor = tensors
    input_lang, target_lang = tokenizers
    # Create batches from a tensorflow dataset, shuffle the data for training
    dataset = tf.data.Dataset.from_tensor_slices((input_tensor, target_tensor)).shuffle(len(input_tensor))
    dataset = dataset.batch(batch_size, drop_remainder=True)

    vocab_input_size = len(input_lang.word_index)+1
    vocab_target_size = len(target_lang.word_index)+1

    # Define the encoder and decoder
    encoder = Encoder(vocab_input_size, embed_dim, encode_units, batch_size)
    decoder = Decoder(vocab_target_size, embed_dim, encode_units, batch_size)

    # Define the optimizer
    optim = tf.keras.optimizers.Adam(learning_rate=lr)

    # Object used to calculate loss
    loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,reduction="none")

    steps_per_epoch = len(input_tensor)//batch_size

    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optim,
                                    encoder=encoder,
                                    decoder=decoder)

    # Now we can train!
    train(num_epochs, steps_per_epoch, lr, batch_size, embed_dim, encode_units)
        
def translate(sentence):

  # Clean the sentence
  sentence = clean_sentence(sentence)

  # The translation result
  result = ""

  # Initalize first encoder state
  encoder_hidden = [tf.zeros((1, encode_units))]
  encoder_out, encoder_hidden = encoder(sentence, encoder_hidden)

  # Feed start token into decoder
  decoder_hidden = encoder_hidden
  current_word = tf.expand_dims([target_lang.word_index['<start>']], 0)

  # Feed the decoder our sentence
  for word_index in range(max_length):

    # Pass input through the decoder
    logits, decoder_hidden, attention_weights = decoder(current_word, decoder_hidden, encoder_out)

    predicted_id = tf.argmax(logits[0]).numpy()

    # Check to see if the sentence is ended
    if target_lang.index_word[predicted_id] == "<end>":
      break

    # Add space between words
    result += target_lang.index_word[predicted_id] + " "

    # Feed the next decoder state
    current_word = tf.expand_dims([predicted_id], 0)

  print(result)