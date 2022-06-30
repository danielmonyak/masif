# Header variables and parameters.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import sys
import importlib
from IPython.core.debugger import set_trace
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model
from default_config.util import *

continue_training = False

params = masif_opts["ligand"]

datadir = '/data02/daniel/masif/datasets/tf2/ligand_site_one'
genPath = os.path.join(datadir, '{}_{}.npy')

train_X = np.load(genPath.format('train', 'X'))
train_y = np.load(genPath.format('train', 'y'))
val_X = np.load(genPath.format('val', 'X'))
val_y = np.load(genPath.format('val', 'y'))

##
def binarize_y(y):
  y[y > 0] = 1
  return tf.constant(y)

train_y = binarize_y(train_y)
val_y = binarize_y(val_y)
##

model = Sequential([
  layers.Dense(1, activation="relu"),
  layers.Flatten(),
  layers.Dense(64, activation="relu"),
  layers.Dense(20, activation='relu'),
  layers.Dense(1)
])

opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits = True)

model.compile(optimizer = opt,
  loss = loss_fn,
  metrics=['binary_accuracy']
)

modelDir = 'kerasModel'
ckpPath = os.path.join(modelDir, 'ckp')

last_epoch = 0
initValThresh = None

if continue_training:
  model.load_weights(ckpPath)
  last_epoch += 13
  initValThresh = 0.91657


gpus = tf.config.experimental.list_logical_devices('GPU')
gpus_str = [g.name for g in gpus]
strategy = tf.distribute.MirroredStrategy(gpus_str[1:])


saveCheckpoints = tf.keras.callbacks.ModelCheckpoint(
  ckpPath,
  monitor = 'val_binary_accuracy',
  save_best_only = True,
  verbose = 1,
  initial_value_threshold = initValThresh
)

dev = '/GPU:3'

num_epochs = 200
#with strategy.scope():
with tf.device(dev):
  history = model.fit(x = train_X, y = train_y,
    epochs = num_epochs - last_epoch,
    validation_data = (val_X, val_y),
    callbacks = [saveCheckpoints],
    verbose = 2,
    use_multiprocessing = False
  )
