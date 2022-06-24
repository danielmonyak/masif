import tensorflow as tf
from tensorflow import keras

class Linear(keras.layers.Layer):
    def __init__(self, units=6, input_dim=3):
        super(Linear, self).__init__()
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_dim, units), dtype="float32"),
            trainable=True)
        
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(units,), dtype="float32"), trainable=True)
    
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

l = Linear(2, 3)

x = tf.constant([[[1, 2, 3], [1, 2, 3]]], dtype = tf.float32)
l(x)

