# Header variables and parameters.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import numpy as np
from IPython.core.debugger import set_trace
import importlib
import sys
from default_config.masif_opts import masif_opts
from tf2.masif_ligand.MaSIF_ligand_TF2 import MaSIF_ligand
from sklearn.metrics import confusion_matrix
import pickle
import tensorflow as tf

#lr = 1e-3
# Try this learning rate after

continue_training = False
dev = '/GPU:3'
cpu = '/CPU:0'

params = masif_opts["ligand"]

datadir = '/data02/daniel/masif/datasets/tf2/masif_ligand'
genPath = os.path.join(datadir, '{}_{}.npy')

train_X = np.load(genPath.format('train', 'X'))
train_y = np.load(genPath.format('train', 'y'))
val_X = np.load(genPath.format('val', 'X'))
val_y = np.load(genPath.format('val', 'y'))

defaultCode = 123.45679

with tf.device(cpu):
  train_X = tf.RaggedTensor.from_tensor(train_X, padding=defaultCode)
  val_X = tf.RaggedTensor.from_tensor(val_X, padding=defaultCode)


modelDir = 'kerasModel'
ckpPath = os.path.join(modelDir, 'ckp')


gpus = tf.config.experimental.list_logical_devices('GPU')
gpus_str = [g.name for g in gpus]
strategy = tf.distribute.MirroredStrategy(gpus_str[2:])

num_epochs = 200
#with tf.device(dev):
with strategy.scope():
  model = MaSIF_ligand(
    params["max_distance"],
    params["n_classes"],
    feat_mask=params["feat_mask"]
  )
  model.compile(optimizer = model.opt,
    loss = model.loss_fn,
    metrics = ['categorical_accuracy']
  )  
  if continue_training:
    model.load_weights(ckpPath)
    last_epoch = 150
    initValThresh = 0
  else:
    last_epoch = 0
    initValThresh = 0

  saveCheckpoints = tf.keras.callbacks.ModelCheckpoint(
    ckpPath,
    monitor = 'val_categorical_accuracy',
    save_best_only = True,
    verbose = 1,
    initial_value_threshold = initValThresh
  )
    
  history = model.fit(x = train_X, y = train_y,
    epochs = num_epochs,
    initial_epoch = last_epoch,
    validation_data = (val_X, val_y),
    callbacks = [saveCheckpoints],
    verbose = 2,
    use_multiprocessing = True
  )

