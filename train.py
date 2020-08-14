import argparse, json
import numpy as np
import tensorflow as tf
from lang_model import Encoder
from lang_model import Decoder

if __name__ == "__main__":

    # Get Command Line Arguments
    parser = argparse.ArgumentParser(description="Shakespeare Translator in TensorFlow")
    parser.add_argument("data", metavar="data", help="Folder containing numpy files of original and translated data", type=str)
    parser.add_argument("params",metavar="params/param_file_name.json",type=str)
    args = parser.parse_args()

    # Hyperparameters from json file
    with open(args.params) as paramfile:
        hyper = json.load(paramfile)

    # load data
    print("Loading data")
    original = np.load(args.data + "/original.npy")
    translation = np.load(args.data + "/translation.npy")

    