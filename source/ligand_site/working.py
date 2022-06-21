# Header variables and parameters.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import sys
from IPython.core.debugger import set_trace
import importlib
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import tensorflow as tf
from default_config.util import *
from tf2.read_ligand_tfrecords import _parse_function
from MaSIF_ligand_site import MaSIF_ligand_site

params = masif_opts["ligand"]
defaultCode = params['defaultCode']
minPockets = params['minPockets']
savedPockets = params['savedPockets']

'''
datadir = '/data02/daniel/masif/datasets/ligand_site'
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
'''
target_pdb = '1RI4_A_'

test_data = tf.data.TFRecordDataset(os.path.join(params["tfrecords_dir"], 'testing_data_sequenceSplit_30.tfrecord')).map(_parse_function)
for i, data_element in enumerate(test_data):
  print(i)
  print(data_element[5])
  if data_element[5] != target_pdb:
    continue
  
  labels = tf.squeeze(data_element[4])
  pocket_points = tf.squeeze(tf.where(labels != 0))
  npoints = pocket_points.shape[0]
  savedPockets_temp = min(savedPockets, npoints)
  
  ##
  pocket_points = tf.random.shuffle(pocket_points)[:savedPockets_temp]
  npoints = savedPockets_temp
  ##
  pocket_empties = tf.squeeze(tf.where(labels == 0))
  empties_sample = tf.random.shuffle(pocket_empties)[:npoints*ratio]
  sample = tf.concat([pocket_points, empties_sample], axis=0)
  
  y = tf.gather(labels, sample)
  
  input_feat = tf.gather(data_element[0], sample)
  rho_coords = tf.gather(tf.expand_dims(data_element[1], -1), sample)
  theta_coords = tf.gather(tf.expand_dims(data_element[2], -1), sample)
  mask = tf.gather(data_element[3], sample)
  
  feed_dict = {
      'input_feat' : input_feat,
      'rho_coords' : rho_coords,
      'theta_coords' : theta_coords,
      'mask' : mask
  }
  
  def helperInner(tsr_key):
      tsr = feed_dict[tsr_key]
      return tf.reshape(tsr, [-1])
  
  key_list = ['input_feat', 'rho_coords', 'theta_coords', 'mask']
  flat_list = list(map(helperInner, key_list))
  X = tf.concat(flat_list, axis = 0)
  
  break

'''
modelDir = 'kerasModel'
ckpPath = os.path.join(modelDir, 'ckp')

model = MaSIF_ligand_site(
  params["max_distance"],
  params["n_classes"],
  feat_mask=params["feat_mask"],
  keep_prob = 1.0
)
model.load_weights(ckpPath)

gen_sample = tf.range(y.shape[1])
sample = tf.stack([gen_sample] * y.shape[0])

dev = '/GPU:3'
with tf.device(dev):
  y_pred = tf.squeeze(model(X, sample))
  y_pred = tf.cast(y_pred > 0.5, dtype=tf.int64)
  y_true = tf.gather(params = y, indices = sample, axis = 1, batch_dims = 1)
'''
'''
acc = balanced_accuracy_score(flatten(y_true), flatten(y_pred))
print('Balanced accuracy: ', round(acc, 2))
'''
