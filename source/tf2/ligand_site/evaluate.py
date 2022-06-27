# Header variables and parameters.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import numpy as np
from IPython.core.debugger import set_trace
import importlib
import sys
from default_config.util import *
from MaSIF_ligand_site import MaSIF_ligand_site
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import tensorflow as tf

params = masif_opts["ligand"]
defaultCode = params['defaultCode']
minPockets = params['minPockets']

datadir = '/data02/daniel/masif/datasets/tf2/ligand_site'
genPath = os.path.join(datadir, '{}_{}.npy')

X = np.load(genPath.format('test', 'X'))
y = np.load(genPath.format('test', 'y'))

##
y[y > 0] = 1
y = tf.constant(y)
##

cpu = '/CPU:0'
with tf.device(cpu):
  X = tf.RaggedTensor.from_tensor(X, padding=defaultCode)
  y = tf.RaggedTensor.from_tensor(y, padding=defaultCode)

modelDir = 'kerasModel'
ckpPath = os.path.join(modelDir, 'ckp')

model = MaSIF_ligand_site(
  params["max_distance"],
  params["n_classes"],
  feat_mask=params["feat_mask"],
  n_conv_layers = 4
)
model.load_weights(ckpPath)

def map_func(row):
  pocket_points = tf.where(row != 0)
  pocket_points = tf.random.shuffle(pocket_points)[:int(minPockets/2)]
  pocket_empties = tf.where(row == 0)
  pocket_empties = tf.random.shuffle(pocket_empties)[:int(minPockets/2)]
  return tf.cast(
    tf.squeeze(
      tf.concat([pocket_points, pocket_empties], axis = 0)
    ),
    dtype=tf.int32
  )

sample = tf.map_fn(fn=map_func, elems = y, fn_output_signature = sampleSpec)

def map_func(row):
  y_pred = row[0]
  y_true = row[1]
  
  mask = tf.boolean_mask(y_pred, y_true)
  overlap = tf.reduce_sum(mask)
  recall = overlap/tf.reduce_sum(y_true)
  precision = overlap/tf.reduce_sum(y_pred)
  specificity = 1 - tf.reduce_mean(tf.cast(tf.boolean_mask(y_pred, 1 - y_true), dtype=tf.float64))
  
  #return (recall, precision)
  return (recall, precision, specificity)

dev = '/GPU:3'
with tf.device(dev):
  y_pred = tf.squeeze(model(X, sample))
  y_pred = tf.cast(y_pred > 0.5, dtype=tf.int64)
  y_true = tf.gather(params = y, indices = sample, axis = 1, batch_dims = 1)
'''  input = tf.stack([y_pred, y_true], axis=1)
  spec_float = tf.TensorSpec(shape=(), dtype=tf.float64)
  #spec_int = tf.TensorSpec(shape=(), dtype=tf.int64)
  recall_tsr, precision_tsr, specificity_tsr = tf.map_fn(fn=map_func, elems = input, fn_output_signature = (spec_float, spec_float, spec_float))

bal_acc = balanced_accuracy_score(flatten(y_true), flatten(y_pred))


getMean = lambda x : tf.reduce_mean(tf.boolean_mask(x, tf.math.is_finite(x)))
recall = getMean(recall_tsr).numpy()
precision = getMean(precision_tsr).numpy()
specificity = getMean(specificity_tsr).numpy()
'''

y_true_all = flatten(y_true)
y_pred_all = flatten(y_pred)
bal_acc = balanced_accuracy_score(y_true_all, y_pred_all)
mask = tf.boolean_mask(y_pred_all, y_true_all)
overlap = tf.reduce_sum(mask)
recall = overlap/tf.reduce_sum(y_true_all)
precision = overlap/tf.reduce_sum(y_pred_all)
specificity = 1 - tf.reduce_mean(tf.cast(tf.boolean_mask(y_pred_all, 1 - y_true_all), dtype=tf.float64))

print('Balanced accuracy:', round(bal_acc, 2))
print('Recall:', round(recall, 2))
print('Precision:', round(precision, 2))
print('Specificity:', round(specificity, 2))
