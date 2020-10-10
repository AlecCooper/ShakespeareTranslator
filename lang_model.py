import tensorflow as tf
from tensorflow.keras import Model

class Encoder(Model):

    def __init__(self, vocab_len, embed_dim, encode_units, batch_size):
        super(Encoder, self).__init__()
        
        # The size of each batch used when minibatch training
        self.batch_size = batch_size

        # The output size after our embedding layer
        self.embed_dim = embed_dim

        # The input dimension to the RNN layers
        self.encode_units = encode_units

        # The embedding layer takes our input,  a vector of positive integers
        # representing indicies and turns them into a dense vector
        self.embedding = tf.keras.layers.Embedding(vocab_len, self.embed_dim, mask_zero=True)

        # The GRU layers
        self.gru_forward = tf.keras.layers.GRU(self.encode_units,return_sequences=True, recurrent_initializer='glorot_uniform')
        self.gru_backward = tf.keras.layers.GRU(self.encode_units,return_sequences=True, return_state=True, go_backwards=True, recurrent_initializer='glorot_uniform')

    def call(self, seq_input, hidden_state):

        # For the first layer we embedd the input into a dense vector
        embed_output = self.embedding(seq_input)

        # 2nd layer, we pass through the GRUs
        forward_output = self.gru_forward(embed_output,initial_state=hidden_state)
        backward_output, hidden_state = self.gru_backward(embed_output,initial_state=hidden_state)

        output = tf.concat([forward_output, backward_output],axis=2)

        return output, hidden_state

    def initalize_hidden(self):
        return tf.zeros((self.batch_size, self.encode_units))

class Attention(tf.keras.layers.Layer):
    
    def __init__(self, units):
        super(Attention, self).__init__()

        # Layers in the MLP used to calculate score
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):

        # Brodacast addition over the time axis
        query = tf.expand_dims(query, 1)

        # Calculate the attention score with a MLP
        score = self.V(tf.nn.tanh(self.W1(query) + self.W2(values)))

        # Softmax calculates the attention weights
        attention_weights = tf.nn.softmax(score, axis=1)

        # Calculate the context vector
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights
        

class Decoder(Model):

    def __init__(self,vocab_len, embed_dim, decode_units, batch_size):
        super(Decoder, self).__init__()

        # The size of each batch used when minibatch training
        self.batch_size = batch_size

        # The input dimension to the RNN layers
        self.decode_units = decode_units

        # The embedding layer takes our input,  a vector of positive integers
        # representing indicies and turns them into a dense vector
        self.embedding = tf.keras.layers.Embedding(vocab_len, embed_dim, mask_zero=True)

        # The GRU layers are the RNNs in our encoder decoder, which are stacked
        # ontop of each other
        self.GRU = tf.keras.layers.GRU(self.decode_units,return_sequences=True,return_state=True)

        # Fully connected layer
        self.fc = tf.keras.layers.Dense(vocab_len)

        # Initalize so we can calculate attention scores and context vectors
        self.attention = Attention(self.decode_units)


    def call(self, seq_input, hidden_state, encoder_output, target_word=None):

        # Create the context vector and weights for the attention mechanism
        context_vector, attention_weights = self.attention(hidden_state, encoder_output)

        # For the first layer we embedd the input into a dense vector
        embed_output = self.embedding(seq_input)

        concat_output = tf.concat([tf.expand_dims(context_vector, 1), embed_output], axis=-1)

        # 2nd layer, we pass through the GRU
        seq_output, hidden_state = self.GRU(concat_output,initial_state=hidden_state)

        # Output passed through the fully connected layer
        output = tf.reshape(seq_output, (-1, seq_output.shape[2]))
        output = self.fc(output)

        return output, hidden_state, attention_weights

    def initalize_hidden(self):
        return tf.zeros((self.batch_size, self.encode_units))

        
