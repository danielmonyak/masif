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

datadirLS = '/data02/daniel/masif/datasets/ligand_site'
#genPath = os.path.join(datadir, '{}_{}_{}.npy')
datadirTF2 = '/data02/daniel/masif/datasets/tf2'
#genPath = os.path.join(datadir, '{}_{}.npy')

dev = '/GPU:1'
train_j = range(4)
with tf.device(dev):
  '''j = 0
  train_X_list = []
  for j in range(2):
    print(j)
    #train_X_temp = np.load(os.path.join(datadirLS, 'train_X_{}.npy'.format(j)))
    train_X_temp = np.load(os.path.join(datadirTF2, 'train_X.npy'))
    print('one')
    train_X_temp = tf.RaggedTensor.from_tensor(train_X_temp, padding=defaultCode)
    print('two')
    train_X_list.append(train_X_temp)
  print('three')
  train_X = tf.concatenate(train_X_list)
  print('four')
  #train_y_raw = np.load(genPath.format('train', 'y'))
  #train_y = tf.RaggedTensor.from_tensor(train_y_raw, padding=defaultCode)'''
  train_X_temp = np.load(os.path.join(datadirTF2, 'train_X.npy'))
  train_X_temp = tf.RaggedTensor.from_tensor(train_X_temp, padding=defaultCode)
  
  print(tf.shape(train_X_temp))
