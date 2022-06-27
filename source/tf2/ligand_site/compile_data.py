# Header variables and parameters.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import sys
import numpy as np
import tensorflow as tf
from default_config.util import *
from tf2.read_ligand_tfrecords import _parse_function

params = masif_opts["ligand"]
defaultCode = params['defaultCode']

datadir = '/data02/daniel/masif/datasets/tf2/ligand_site/split'
all_files = np.array(os.listdir(datadir))

outDir = '/data02/daniel/masif/datasets/tf2/ligand_site'
genPathOut = os.path.join(outDir, '{}_{}.npy')

dev = '/GPU:1'

#with tf.device(dev):
for dataset in ['train', 'val', 'test']:
  temp_X_files = all_files[np.char.startswith(all_files, dataset + '_X')]

  print(dataset)
  X_list = []
  y_list = []
  len_list = []
  for j, X_file in enumerate(temp_X_files):
    print(j)
    y_file = X_file.replace('X', 'y')
    X = np.load(os.path.join(datadir, X_file))
    y = np.load(os.path.join(datadir, y_file))
    tempLen = tf.shape(X)[1]
    X_list.append(X)
    y_list.append(y)
    len_list.append(tempLen)

  #### padding X_list elements so that they all have the same dimension-2 length
  maxLen = max(len_list)
  for i, X in enumerate(X_list):
    paddings = tf.constant([[0, 0], [0, (maxLen - len_list[i]).numpy()]])
    X_list[i] = tf.pad(X, paddings, constant_values=defaultCode)
  ####

  X = tf.concat(X_list, axis=0)
  y = tf.concat(y_list, axis=0)
  np.save(genPathOut.format(dataset, 'X'), X)
  np.save(genPathOut.format(dataset, 'y'), y)

print('Finished!')
