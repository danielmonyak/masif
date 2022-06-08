# Header variables and parameters.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import numpy as np
from IPython.core.debugger import set_trace
import importlib
import sys
from default_config.masif_opts import masif_opts
from MaSIF_ligand_TF2 import MaSIF_ligand
from sklearn.metrics import confusion_matrix
import pickle
import tensorflow as tf

continue_training = False


datadir = 'datasets/'
train_X = np.load(datadir + 'train_X.npy')
train_y = np.load(datadir + 'train_y.npy')
val_X = np.load(datadir + 'val_X.npy')
val_y = np.load(datadir + 'val_y.npy')
test_X = np.load(datadir + 'test_X.npy')
test_y = np.load(datadir + 'test_y.npy')


params = masif_opts["ligand"]

model = MaSIF_ligand(
  params["max_distance"],
  params["n_classes"],
  feat_mask=params["feat_mask"],
  keep_prob = 0.75
)
model.compile(optimizer = model.opt,
  loss = model.loss_fn,
  metrics=['accuracy']
)

modelDir = 'kerasModel'
ckpPath = os.path.join(modelDir, 'ckp')

last_epoch = 0
initValThresh = None

if continue_training:
  model.load_weights(ckpPath)
  last_epoch += 22
  initValThresh = 0.62606


gpus = tf.config.experimental.list_logical_devices('GPU')
gpus_str = [g.name for g in gpus]
strategy = tf.distribute.MirroredStrategy(gpus_str[1:])


saveCheckpoints = tf.keras.callbacks.ModelCheckpoint(
  ckpPath,
  monitor = 'val_accuracy',
  save_best_only = True,
  verbose = 1,
  initial_value_threshold = initValThresh
)

num_epochs = 100
with strategy.scope():
  history = model.fit(x = train_X, y = train_y,
    epochs = num_epochs - last_epoch,
    validation_data = (val_X, val_y),
    callbacks = [saveCheckpoints],
    verbose = 2,
    use_multiprocessing = True
  )

with open(os.path.join(modelDir, 'train_history'), 'wb') as file_pi:
  pickle.dump(history.history, file_pi)
