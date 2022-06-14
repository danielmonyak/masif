# Header variables and parameters.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import numpy as np
import sys
from default_config.masif_opts import masif_opts
from tf2.read_ligand_tfrecords import _parse_function
import tensorflow as tf

params = masif_opts["ligand"]

datadir = '/data02/daniel/masif/datasets/ligand_site/split'
genPathIn = os.path.join(datadir, '{}_{}_{}.npy')

outDir = '/data02/daniel/masif/datasets/ligand_site'
genPathOut = os.path.join(outDir, '{}_{}.npy')

dev = '/GPU:1'
train_j = range(10)

numFiles_dict = {'train' : 10, 'val' : 2, 'test' : 3}

for dataset in numFiles_dict.keys():
  print(dataset)
  X_list = []
  y_list = []
  for j in range(numFiles_dict[dataset]):
    print(j)
    X_list.append(np.load(genPathIn.format(dataset, 'X', j)))
    y_list.append(np.load(genPathIn.format(dataset, 'y', j)))
  X = tf.concat(X_list, axis=0)
  y = tf.concat(y_list, axis=0)
  np.save(genPathOut.format(dataset, 'X'), X)
  np.save(genPathOut.format(dataset, 'y'), y)
