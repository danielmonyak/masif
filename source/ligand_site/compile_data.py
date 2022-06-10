# Header variables and parameters.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import numpy as np
import sys
from default_config.masif_opts import masif_opts
from tf2.read_ligand_tfrecords import _parse_function
import tensorflow as tf

params = masif_opts["ligand"]
defaultCode = params['defaultCode']

#datadir = '/data02/daniel/masif/datasets/ligand_site'
#genPath = os.path.join(datadir, '{}_{}_{}.npy')
datadir = '/data02/daniel/masif/datasets/tf2'
genPath = os.path.join(datadir, '{}_{}.npy')

dev = '/GPU:1'
with tf.device(dev):
  j = 4
  print('one')
  train_X_raw = np.load(genPath.format('train', 'X'))
  print('two')
  train_y_raw = np.load(genPath.format('train', 'y'))
  print('three')
  train_X = tf.RaggedTensor.from_tensor(train_X_raw, padding=defaultCode)
  print('four')
  #train_y = tf.RaggedTensor.from_tensor(train_y_raw, padding=defaultCode)
  
