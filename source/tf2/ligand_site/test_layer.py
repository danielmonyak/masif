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

a = tf.constant([[1, 2, 3], [1, 2, 3]], dtype = tf.float32)
x = tf.stack([a, a*2, a*10], axis=0)
y = tf.constant([[1, 0], [1,1], [0, 0]])
#y = tf.expand_dims(y, axis=-1)


lin = Linear(2, 3)
model = keras.Sequential(
    [lin, keras.layers.Dense(1)]
)

model.compile(optimizer = 'adam', loss = tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['binary_accuracy'])

model.fit(x,y)
