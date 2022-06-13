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

datadir = '/data02/daniel/masif/datasets/ligand_site'
genPath = os.path.join(datadir, '{}_{}_{}.npy')
#datadirTF2 = '/data02/daniel/masif/datasets/tf2'
#genPath = os.path.join(datadir, '{}_{}.npy')

dev = '/GPU:1'
train_j = range(10)

#train_X_temp = np.load(os.path.join(datadirLS, 'train_X_{}.npy'.format(0)))
#train_X_temp = tf.RaggedTensor.from_tensor(train_X_temp, padding=defaultCode)

train_X_list = []
for j in train_j:
  print(j)
  train_X_list.append(np.load(genPath.format('train', 'X', j)))

train_X = tf.concat(train_X_list, axis=0)
