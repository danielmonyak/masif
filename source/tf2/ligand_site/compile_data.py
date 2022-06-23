# Header variables and parameters.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import sys
import numpy as np
import tensorflow as tf
from default_config.util import *
from tf2.read_ligand_tfrecords import _parse_function

params = masif_opts["ligand"]

datadir = '/data02/daniel/masif/datasets/tf2/ligand_site/split'
all_files = np.array(os.listdir(datadir))

outDir = '/data02/daniel/masif/datasets/tf2/ligand_site'
genPathOut = os.path.join(outDir, '{}_{}.npy')

dev = '/GPU:1'
train_j = range(10)

#numFiles_dict = {'train' : 10, 'val' : 2, 'test' : 3}

with tf.device('/GPU:1'):
for dataset in ['train', 'val', 'test']:
  temp_X_files = all_files[np.char.startswith(all_files, dataset + '_X')]
  
  print(dataset)
  X_list = []
  y_list = []
  for j, X_file in enumerate(temp_X_files):
    print(j)
    y_file = X_file.replace('X', 'y')
    
    X_list.append(np.load(os.path.join(datadir, X_file)))
    y_list.append(np.load(os.path.join(datadir, y_file)))
  X = tf.concat(X_list, axis=0)
  y = tf.concat(y_list, axis=0)
  np.save(genPathOut.format(dataset, 'X'), X)
  np.save(genPathOut.format(dataset, 'y'), y)

print('Finished!')
