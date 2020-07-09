import tensorflow as tf
from tensorflow.keras import Model

class Encoder(Model):

    def __init__(self, vocab_len, embed_dim, batch_size):
        super(Encoder, self).__init__()
        
        # The size of each batch used when minibatch training
        self.batch_size = batch_size

        # The embedding layer takes our input,  a vector of positive integers
        # representing indicies and turns them into a dense vector
        self.embedding = tf.keras.layers.Embedding(vocab_len, embed_dim)

        # The lstm layers are the RNNs in our encoder decoder, which are stacked
        # ontop of each other
        self.lstm = tf.keras.layers.LSTM(self)
    

    def call(self):

        pass


class Decoder(Model):

    def __init__(self,vocab_len, embed_dim, batch_size):
        super(Encoder, self).__init__()

        # The size of each batch used when minibatch training
        self.batch_size = batch_size

        # The embedding layer takes our input,  a vector of positive integers
        # representing indicies and turns them into a dense vector
        self.embedding = tf.keras.layers.Embedding(vocab_len, embed_dim)

        # The lstm layers are the RNNs in our encoder decoder, which are stacked
        # ontop of each other
        self.lstm = tf.keras.layers.LSTM(self)

    def call(self):

        pass


class Attention(tf.keras.layers.Layer):
    def __init__(self):
        super(Attention, self).__init__()
        

