# Header variables and parameters.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import numpy as np
from IPython.core.debugger import set_trace
import importlib
import sys
from default_config.masif_opts import masif_opts
from MaSIF_ligand_TF2 import MaSIf_ligand
from sklearn.metrics import confusion_matrix
import tensorflow as tf

datadir = 'datasets/'
train_X = np.load(datadir + 'train_X.npy')
train_y = np.load(datadir + 'train_y.npy')
val_X = np.load(datadir + 'val_X.npy')
val_y = np.load(datadir + 'val_X.npy')
test_X = np.load(datadir + 'test_X.npy')
test_y = np.load(datadir + 'test_y.npy')

modelDir = 'kerasModel'
# model = tf.keras.models.load_model(modelDir)
model = MaSIF_ligand(
  params["max_distance"],
  params["n_classes"],
  idx_gpu="/gpu:0",
  feat_mask=params["feat_mask"],
  costfun=params["costfun"]
)

gpus = tf.compat.v1.config.experimental.list_logical_devices('GPU')
gpus_str = [g.name for g in gpus]
strategy = tf.distribute.MirroredStrategy(gpus_str)


saveCheckpoints = tf.keras.callbacks.ModelCheckpoint(
  modelDir,
  monitor = 'val_accuracy',
  save_best_only = True,
  verbose = 1
)

num_epochs = 100
with strategy.scope():
  history = model.fit(x = train_X, y = train_y,
    epochs = num_epochs,
    validation_data = (val_X, val_y),
    callbacks = [saveCheckpoints],
    verbose = 2,
    use_multiprocessing = True
  )
  model.evaluate(x_test,  y_test_encoded, verbose=2)

#model.save(modelDir)
with open(modelDir + '/train_history', 'wb') as file_pi:
  pickle.dump(history.history, file_pi)
