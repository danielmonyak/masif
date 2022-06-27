# Header variables and parameters.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import sys
import importlib
from IPython.core.debugger import set_trace
import pickle
import numpy as np
import tensorflow as tf
from default_config.util import *
from tf2.ligand_site.MaSIF_ligand_site import MaSIF_ligand_site

continue_training = False

params = masif_opts["ligand"]
defaultCode = params['defaultCode']

datadir = '/data02/daniel/masif/datasets/tf2/ligand_site'
genPath = os.path.join(datadir, '{}_{}.npy')

train_X = np.load(genPath.format('train', 'X'))
train_y = np.load(genPath.format('train', 'y'))
val_X = np.load(genPath.format('val', 'X'))
val_y = np.load(genPath.format('val', 'y'))

if defaultCode > 0:
  sys.exit("defaultCode will be erased...")

##
def binarize_y(y):
  y[y > 0] = 1
  return tf.constant(y)

train_y = binarize_y(train_y)
val_y = binarize_y(val_y)
##

cpu = '/CPU:0'
with tf.device(cpu):
  train_X = tf.RaggedTensor.from_tensor(train_X, padding=defaultCode)
  train_y = tf.RaggedTensor.from_tensor(train_y, padding=defaultCode)
  val_X = tf.RaggedTensor.from_tensor(val_X, padding=defaultCode)
  val_y = tf.RaggedTensor.from_tensor(val_y, padding=defaultCode)



model = MaSIF_ligand_site(
  params["max_distance"],
  params["n_classes"],
  feat_mask=params["feat_mask"],
  n_conv_layers = 4
)
model.compile(optimizer = model.opt,
  loss = model.loss_fn,
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
