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

#lastEpoch = 4
#epochSize = 25

params = masif_opts["ligand"]
defaultCode = params['defaultCode']
n_classes = params['n_classes']
minPockets = params['minPockets']

outdir = '/data02/daniel/masif/datasets/ligand_site'
genOutPath = os.path.join(outdir, '{}_{}.npy')

def helper(feed_dict):
    def helperInner(tsr_key):
        tsr = feed_dict[tsr_key]
        return tf.reshape(tsr, [-1])
    key_list = ['input_feat', 'rho_coords', 'theta_coords', 'mask']
    flat_list = list(map(helperInner, key_list))
    return tf.concat(flat_list, axis = 0)

def compile_and_save(feed_list, y_list, dataset):
    tsr_list = list(map(helper, feed_list))
    X = tf.ragged.stack(tsr_list).to_tensor(default_value = defaultCode)
    y = tf.ragged.stack(y_list).to_tensor(default_value = defaultCode)
    np.save(genOutPath.format(dataset, 'X'), X)
    np.save(genOutPath.format(dataset, 'y'), y)

dataset_list = {'train' : "training_data_sequenceSplit_30.tfrecord", 'val' : "validation_data_sequenceSplit_30.tfrecord", 'test' : "testing_data_sequenceSplit_30.tfrecord"}
#dataset_list = {'train' : "training_data_sequenceSplit_30.tfrecord"}

gpus = tf.config.experimental.list_logical_devices('GPU')
dev = '/GPU:1'
#tf.config.experimental.set_memory_growth(gpus, True)
gpus_str = [g.name for g in gpus]
strategy = tf.distribute.MirroredStrategy(gpus_str[1:])

#with strategy.scope():
with tf.device(dev):
    for dataset in dataset_list.keys():
        i = 0
        #j = lastEpoch

        feed_list = []
        y_list = []

        temp_data = tf.data.TFRecordDataset(os.path.join(params["tfrecords_dir"], dataset_list[dataset])).map(_parse_function)
        for data_element in temp_data:
            '''if i < j*epochSize:
                i += 1
                continue'''
            
            print('{} record {}'.format(dataset, i))
            
            labels_raw = data_element[4]
            n_ligands = labels_raw.shape[1]
            if n_ligands > 1:
                print('More than one ligand, check this out...')
                continue
            
            #with tf.device(dev):
            #print('a:', process_time())
            labels = tf.squeeze(labels_raw)
            
            #print('b:', process_time())
            y_list.append(labels)
            
            #print('c:', process_time())
            pocket_points = tf.squeeze(tf.where(labels != 0))
            npoints = pocket_points.shape[0]
            if npoints < minPockets:
                continue
            
            #print('d:', process_time())
            pocket_empties = tf.squeeze(tf.where(labels == 0))
            
            #print('e:', process_time())
            empties_sample = tf.random.shuffle(pocket_empties)[:npoints*4]
            
            #print('f:', process_time())
            sample = tf.concat([pocket_points, empties_sample], axis=0)
            
            #one_hot_labels = tf.one_hot(tf.squeeze(labels) - 1, n_classes)
            
        #with tf.device(gpus_str[0]):
            input_feat = tf.gather(data_element[0], sample)
        #with tf.device(gpus_str[1]):
            rho_coords = tf.gather(tf.expand_dims(data_element[1], -1), sample)
        #with tf.device(gpus_str[2]):
            theta_coords = tf.gather(tf.expand_dims(data_element[2], -1), sample)
        #with tf.device(gpus_str[3]):
            mask = tf.gather(data_element[3], sample)
        #print('g:', process_time())
            
            feed_dict = {
                'input_feat' : input_feat,
                'rho_coords' : rho_coords,
                'theta_coords' : theta_coords,
                'mask' : mask
            }
            
            #print('h:', process_time())
            feed_list.append(feed_dict)
            
            #print('i:', process_time())
            i += 1
            
        '''if i % epochSize == 0:
            #with tf.device(dev):
            compile_and_save(feed_list, y_list, j)
            feed_list = []
            y_list = []
            j += 1'''
            
            if i == 10:
                break
        
   # with tf.device(dev):
        compile_and_save(feed_list, y_list, dataset)

print('Finished!')
