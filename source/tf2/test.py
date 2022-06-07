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

#
datadir = 'datasets/'
train_x = np.load(datadir + 'train_X.npy')
train_y = np.load(datadir + 'train_y.npy')
#

model = tf.keras.models.Sequential([
  tf.keras.layers.InputLayer([32, 5, 80]),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(7, activation="softmax")
])

loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    
#loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(train_x, train_y, epochs=10, verbose=2)
