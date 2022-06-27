# Header variables and parameters.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import sys
import importlib
from IPython.core.debugger import set_trace
import numpy as np
import tensorflow as tf
from default_config.util import *
from tf2.read_ligand_tfrecords import _parse_function

epochSize = 100

next_epoch = 0

params = masif_opts["ligand"]
defaultCode = params['defaultCode']
minPockets = params['minPockets']
savedPockets = params['savedPockets']
#empty_pocket_ratio = params['empty_pocket_ratio']
empty_pocket_ratio = 1

outdir = '/data02/daniel/masif/datasets/tf2/ligand_site/split'
genOutPath = os.path.join(outdir, '{}_{}.npy')

if not os.path.exists(outdir):
    os.mkdir(outdir)

def helper(feed_dict):
    def helperInner(tsr_key):
        tsr = feed_dict[tsr_key]
        return flatten(tsr)
    key_list = ['input_feat', 'rho_coords', 'theta_coords', 'mask']
    flat_list = list(map(helperInner, key_list))
    return tf.concat(flat_list, axis = 0)

def compile_and_save(feed_list, y_list, dataset, j):
    tsr_list = list(map(helper, feed_list))
    with tf.device('/CPU:0'):
        X = tf.ragged.stack(tsr_list).to_tensor(default_value = defaultCode)
        y = tf.ragged.stack(y_list).to_tensor(default_value = defaultCode)
    np.save(genOutPath.format(dataset, 'X_{}'.format(j)), X)
    np.save(genOutPath.format(dataset, 'y_{}'.format(j)), y)

dataset_list = {'train' : "training_data_sequenceSplit_30.tfrecord", 'val' : "validation_data_sequenceSplit_30.tfrecord", 'test' : "testing_data_sequenceSplit_30.tfrecord"}

#gpus = tf.config.list_logical_devices('GPU')
gpus = tf.config.list_physical_devices('GPU')
#strategy = tf.distribute.MirroredStrategy(gpus_str[1:])

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

dev = '/GPU:1'
with tf.device(dev):
    #for dataset in dataset_list.keys():
    for dataset in ['val', 'test']:
        i = 0
        j = next_epoch

        feed_list = []
        y_list = []

        temp_data = tf.data.TFRecordDataset(os.path.join(params["tfrecords_dir"], dataset_list[dataset])).map(_parse_function)
        for data_element in temp_data:
            if i < j * epochSize:
                print('Skipping {} record {}'.format(dataset, i))
                i += 1
                continue

            print('{} record {}'.format(dataset, i))

            labels_raw = data_element[4]
            n_ligands = labels_raw.shape[1]
            if n_ligands > 1:
                print('More than one ligand, check this out...')
                continue
            
            
            labels = tf.squeeze(labels_raw)
            pocket_points = tf.squeeze(tf.where(labels != 0))
            npoints = tf.shape(pocket_points)[0]
            if npoints < minPockets:
                continue
            
            savedPockets_temp = min(savedPockets, npoints)
            
            ##
            #pocket_points = tf.random.shuffle(pocket_points)[:savedPockets_temp]
            #npoints = tf.shape(pocket_points)[0]
            ##
            
            pocket_empties = tf.squeeze(tf.where(labels == 0))
            empties_sample = tf.random.shuffle(pocket_empties)[:npoints * empty_pocket_ratio]
            sample = tf.concat([pocket_points, empties_sample], axis=0)
            
            y_list.append(tf.gather(labels, sample))
            
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
            
            feed_list.append(feed_dict)
            i += 1
            
            if i % epochSize == 0:
                compile_and_save(feed_list, y_list, dataset, j)
                feed_list = []
                y_list = []
                j += 1
        
        compile_and_save(feed_list, y_list, dataset, j)


print('Finished!')
