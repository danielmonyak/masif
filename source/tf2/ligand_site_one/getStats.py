import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import sys
import importlib
from IPython.core.debugger import set_trace
import numpy as np
import tensorflow as tf
from default_config.util import *
from tf2.read_ligand_tfrecords import _parse_function
import random

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

dev = '/GPU:1'
cpu = '/CPU:0'

params = masif_opts["ligand"]

dataset_list = {'train' : "training_data_sequenceSplit_30.tfrecord", 'val' : "validation_data_sequenceSplit_30.tfrecord", 'test' : "testing_data_sequenceSplit_30.tfrecord"}
getData = lambda dataset : tf.data.TFRecordDataset(os.path.join(params["tfrecords_dir"], dataset_list[dataset])).map(_parse_function)
train_data = getData('train')

def goodLabel(labels):
    n_ligands = labels.shape[1]
    if n_ligands > 1:
        return False
    
    pocket_points = tf.where(labels != 0)
    npoints = tf.shape(pocket_points)[0]
    if npoints < minPockets:
        return False
    
    return True

n_pockets_list = []
total_pts_list = []

with tf.device(dev):    
    train_iterator = iter(train_data)
    i = 0
    while True:
        try:
            print(i)
            i += 1
            data_element = train_iterator.get_next()
        except:
            break

        labels = data_element[4]
        if not goodLabel(labels):
            continue

        y = tf.cast(labels > 0, dtype=tf.int32)
        
        total_pts_list.append(labels.shape[0])
        n_pockets_list.append(np.sum(labels > 0))

total_pts_arr = np.array(total_pts_list)
n_pockets_arr = np.array(n_pockets_list)
pockets_frac_arr = n_pockets_arr/total_pts_arr

total_pts_mean = np.mean(total_pts_arr)
n_pockets_mean = np.mean(n_pockets_arr)
pockets_frac_mean = np.mean(pockets_frac_arr)

with open('stats.txt', 'w') as f:
    f.write(f'Average number of total points: {total_pts_mean}\n')
    f.write(f'Average number of pocket points: {n_pockets_mean}\n')
    f.write(f'Average fraction of points that are pocket points: {round(pockets_frac_mean, 3)}\n')

