import numpy as np
from random import randint

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


low_x = np.array(list(map(lambda foo : randint(0, 100), range(500))))
low_y = np.zeros(500)

high_x = np.array(list(map(lambda foo : randint(150, 1000), range(500))))
high_y = np.ones(500)

train_x = np.concatenate( [low_x, high_x] ).reshape(  (low_x.shape[0] + high_x.shape[0], 1)  )
train_y = np.concatenate( [low_y, high_y] )




model = tf.keras.models.Sequential([
  tf.keras.layers.InputLayer(1),
  tf.keras.layers.Dense(2, activation="sigmoid")
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(train_x, train_y, epochs=100, verbose=2)
