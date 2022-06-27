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
#for dataset in ['train', 'val', 'test']:
for dataset in ['val', 'test']:
  temp_X_files = all_files[np.char.startswith(all_files, dataset + '_X')]
  
  print(dataset)
  X_list = []
  y_list = []
  lens_list_X = []
  lens_list_y = []
  for j, X_file in enumerate(temp_X_files):
    print(j)
    y_file = X_file.replace('X', 'y')
    X = np.load(os.path.join(datadir, X_file))
    y = np.load(os.path.join(datadir, y_file))
    X_list.append(X)
    y_list.append(y)
    lens_list_X.append(tf.shape(X)[1])
    lens_list_y.append(tf.shape(y)[1])
  
  #### padding X_list elements so that they all have the same dimension-2 length
  maxLen_X = max(lens_list_X)
  maxLen_y = max(lens_list_y)
  for i, X in enumerate(X_list):
    y = y_list[i]
    paddings_X = tf.constant([[0, 0], [0, (maxLen_X - lens_list_X[i]).numpy()]])
    paddings_y = tf.constant([[0, 0], [0, (maxLen_y - lens_list_y[i]).numpy()]])
    X_list[i] = tf.pad(X, paddings_X, constant_values=defaultCode)
    y_list[i] = tf.pad(y, paddings_y, constant_values=defaultCode)
    ####
  
  X = tf.concat(X_list, axis=0)
  y = tf.concat(y_list, axis=0)
  np.save(genPathOut.format(dataset, 'X'), X)
  np.save(genPathOut.format(dataset, 'y'), y)

print('Finished!')
