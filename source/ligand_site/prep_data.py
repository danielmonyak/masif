# Header variables and parameters.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 


import numpy as np
from IPython.core.debugger import set_trace
import importlib
import sys
from default_config.masif_opts import masif_opts
from tf2.read_ligand_tfrecords import _parse_function
import tensorflow as tf
from time import process_time

epochSize = 200

ratio = 1
savedPockets = 64
epochSize = 50


params = masif_opts["ligand"]
defaultCode = params['defaultCode']
n_classes = params['n_classes']
minPockets = params['minPockets']

outdir = '/data02/daniel/masif/datasets/ligand_site'
outdir = '.'
genOutPath = os.path.join(outdir, '{}_{}.npy')

def helper(feed_dict):
    def helperInner(tsr_key):
        tsr = feed_dict[tsr_key]
        return tf.reshape(tsr, [-1])
    key_list = ['input_feat', 'rho_coords', 'theta_coords', 'mask']
    flat_list = list(map(helperInner, key_list))
    return tf.concat(flat_list, axis = 0)

def compile_and_save(feed_list, y_list, dataset, j):
    tsr_list = list(map(helper, feed_list))
    X = tf.ragged.stack(tsr_list).to_tensor(default_value = defaultCode)
    y = tf.ragged.stack(y_list).to_tensor(default_value = defaultCode)
    np.save(genOutPath.format(dataset, 'X_{}'.format(j)), X)
    np.save(genOutPath.format(dataset, 'y_{}'.format(j)), y)

#dataset_list = {'train' : "training_data_sequenceSplit_30.tfrecord", 'val' : "validation_data_sequenceSplit_30.tfrecord", 'test' : "testing_data_sequenceSplit_30.tfrecord"}
dataset_list = {'train' : "training_data_sequenceSplit_30.tfrecord"}

dev = '/GPU:1'
'''gpus = tf.config.experimental.list_logical_devices('GPU')
gpus_str = [g.name for g in gpus]
strategy = tf.distribute.MirroredStrategy(gpus_str[1:])'''
#tf.config.experimental.set_memory_growth(gpus, True)

#with strategy.scope():
with tf.device(dev):
    for dataset in dataset_list.keys():
        i = 0
        j = 0

        feed_list = []
        y_list = []

        temp_data = tf.data.TFRecordDataset(os.path.join(params["tfrecords_dir"], dataset_list[dataset])).map(_parse_function)
        for data_element in temp_data:
            print('{} record {}'.format(dataset, i))
            
            labels_raw = data_element[4]
            n_ligands = labels_raw.shape[1]
            if n_ligands > 1:
                print('More than one ligand, check this out...')
                continue
            
            #print('a:', process_time())
            labels = tf.squeeze(labels_raw)
            
            #print('b:', process_time())
            
            #print('c:', process_time())
            pocket_points = tf.squeeze(tf.where(labels != 0))
            npoints = pocket_points.shape[0]
            if npoints < minPockets:
                continue
            
            ##
            pocket_points = tf.random.shuffle(pocket_points)[:savedPockets]
            npoints = savedPockets
            ##
            pocket_empties = tf.squeeze(tf.where(labels == 0))
            empties_sample = tf.random.shuffle(pocket_empties)[:npoints*ratio]
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
            
            #print('h:', process_time())
            feed_list.append(feed_dict)
            i += 1
            
            #print('i:', process_time())
            if i % epochSize == 0:
                compile_and_save(feed_list, y_list, dataset, j)
                break
                feed_list = []
                y_list = []
                j += 1
        
        compile_and_save(feed_list, y_list, dataset, j)
    

print('Finished!')
