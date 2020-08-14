import argparse, json
import numpy as np
import tensorflow as tf
from lang_model import Encoder
from lang_model import Decoder

# The loss function we use to calculate training loss
def loss_func(actual, pred, loss_obj):

    # Calculate the cross entropy with our loss object
    loss = loss_obj(actual, pred)

    return loss

def train(num_epochs, batch_size, lr, original, translation):

    # Define the optimizer
    optim = tf.keras.optimizers.Adam(learning_rate=lr)

    # Object used to calculate loss
    loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,reduction=None)

    # Training loop
    for epoch in range(1,num_epochs+1):

        
        # Loop through our minibatches
        for batch in range(int(len(original)/batch_size)):

            pass

        pass

if __name__ == "__main__":

    # Get Command Line Arguments
    parser = argparse.ArgumentParser(description="Shakespeare Translator in TensorFlow")
    parser.add_argument("data", metavar="data_dir", help="Folder containing numpy files of original and translated data", type=str)
    parser.add_argument("params",metavar="param_file.json",help="location of hyperparamater json", type=str)
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

    # Train the model
    train(num_epochs, batch_size, lr, original, translation)

    