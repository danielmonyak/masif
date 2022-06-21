# Header variables and parameters.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import numpy as np
from IPython.core.debugger import set_trace
import importlib
import sys
from default_config.masif_opts import masif_opts
from MaSIF_ligand_site import MaSIF_ligand_site
from sklearn.metrics import balanced_accuracy_score
import tensorflow as tf

params = masif_opts["ligand"]
defaultCode = params['defaultCode']

datadir = '/data02/daniel/masif/datasets/ligand_site'
genPath = os.path.join(datadir, '{}_{}.npy')

test_X = np.load(genPath.format('test', 'X'))
test_y = np.load(genPath.format('test', 'y'))

##
def binarize_y(y):
  y[y > 0] = 1
  return tf.constant(y)

test_y = binarize_y(test_y)
##

modelDir = 'kerasModel'
ckpPath = os.path.join(modelDir, 'ckp')

cpu = '/CPU:0'
with tf.device(cpu):
  test_X = tf.RaggedTensor.from_tensor(test_X, padding=defaultCode)
  test_y = tf.RaggedTensor.from_tensor(test_y, padding=defaultCode)

model = MaSIF_ligand_site(
  params["max_distance"],
  params["n_classes"],
  feat_mask=params["feat_mask"],
  keep_prob = 1.0
)
model.compile(optimizer = ligand_site_model.opt,
  loss = ligand_site_model.loss_fn,
  metrics=['accuracy']
)
model.load_weights(ckpPath)

dev = '/GPU:3'
with tf.device(dev):
  y_pred = model(test_X)
'''
balanced_acc = balanced_accuracy_score(test_y, y_pred)
print('Balanced accuracy: ', round(balanced_acc, 2))
'''
