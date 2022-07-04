# Header variables and parameters.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import numpy as np
from IPython.core.debugger import set_trace
import importlib
import sys
from default_config.util import *
from MaSIF_ligand_site import MaSIF_ligand_site
from sklearn.metrics import accuracy_score
import tensorflow as tf

params = masif_opts["ligand"]
defaultCode = params['defaultCode']


modelDir = 'kerasModel'
modelPath = os.path.join(modelDir, 'savedModel')
model = tf.keras.models.load_model(modelPath)

datadir = '/data02/daniel/masif/datasets/tf2'
test_X_raw = np.load(os.path.join(datadir, 'test_X.npy'))
test_y = np.load(os.path.join(datadir, 'test_y.npy'))

defaultCode = 123.45679

with tf.device('/CPU:0'):
  test_X = tf.RaggedTensor.from_tensor(test_X_raw, padding=defaultCode)

gpus = tf.config.experimental.list_logical_devices('GPU')
gpus_str = [g.name for g in gpus]
strategy = tf.distribute.MirroredStrategy(gpus_str[1:])

dev = '/GPU:3'

with strategy.scope():
  print('train')
  train_res = model.evaluate(train_X, train_y)
  print('val')
  val_res = model.evaluate(val_X, val_y)
  print('test')
  test_res = model.evaluate(test_X, test_y)
  y_pred_probs = model.predict(test_X)

y_true = test_y.argmax(axis = 1)
y_pred = y_pred_probs.argmax(axis = 1)

balanced_acc = balanced_accuracy_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_pred_probs, multi_class = 'ovr', labels = np.arange(7))

print('Balanced accuracy:', balanced_acc)
print('ROC AUC:', roc_auc)
