import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Softmax

class FeatureAttention(Layer):
    """
    Input (Feature) Attention: Weights the importance of different features (CPU, Latency, etc.)
    at each individual time step.
    """
    def __init__(self, **kwargs):
        super(FeatureAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="feat_att_weight", shape=(input_shape[-1], input_shape[-1]),
                                 initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="feat_att_bias", shape=(input_shape[-1],),
                                 initializer="zeros", trainable=True)
        super(FeatureAttention, self).build(input_shape)

    def call(self, x):
        # x shape: (batch, seq_len, features)
        score = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        prob = tf.nn.softmax(score, axis=-1)
        return x * prob

class TemporalAttention(Layer):
    """
    Temporal Attention: Weights the importance of specific hidden states across the
    time dimension after BiLSTM/TCN processing.
    """
    def __init__(self, **kwargs):
        super(TemporalAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="temp_att_weight", shape=(input_shape[-1], 1),
                                 initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="temp_att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros", trainable=True)
        super(TemporalAttention, self).build(input_shape)

    def call(self, x):
        # x shape: (batch, seq_len, hidden_dim)
        e = tf.squeeze(tf.nn.tanh(tf.matmul(x, self.W) + self.b), axis=-1)
        a = tf.nn.softmax(e, axis=-1)
        a = tf.expand_dims(a, axis=-1)
        output = x * a
        return tf.reduce_sum(output, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])
