# Header variables and parameters.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 


import numpy as np
from IPython.core.debugger import set_trace
import importlib
import sys
from default_config.util import *
from tf2.read_ligand_tfrecords import _parse_function
import tensorflow as tf

params = masif_opts["ligand"]
defaultCode = params['defaultCode']
minPockets = params['minPockets']

outdir = '/data02/daniel/masif/datasets/tf2/masif_ligand'
genOutPath = os.path.join(outdir, '{}_{}.npy')

def helper(feed_dict):
    def helperInner(tsr_key):
        tsr = feed_dict[tsr_key]
        return flatten(tsr)
    flat_list = list(map(helperInner, data_order))
    return tf.concat(flat_list, axis = 0)

def compile_and_save(feed_list, y_list, dataset):
    tsr_list = list(map(helper, feed_list))
    with tf.device('/CPU:0'):
        X = tf.ragged.stack(tsr_list).to_tensor(default_value = defaultCode)
        y = tf.stack(y_list, axis = 0)
    np.save(genOutPath.format(dataset, 'X'), X)
    np.save(genOutPath.format(dataset, 'y'), y)


dataset_list = {'train' : "training_data_sequenceSplit_30.tfrecord", 'val' : "validation_data_sequenceSplit_30.tfrecord", 'test' : "testing_data_sequenceSplit_30.tfrecord"}

for dataset in dataset_list.keys():
    i = 0
    
    feed_list = []
    y_list = []

    temp_data = tf.data.TFRecordDataset(os.path.join(params["tfrecords_dir"], dataset_list[dataset])).map(_parse_function)
    temp_iterator = iter(temp_data)
    for data_element in temp_data:
        data_element = temp_iterator.get_next_as_optional()
        if not data_element:
            
        
        print('{} record {}'.format(dataset, i))
        
        labels = data_element[4]
        n_ligands = labels.shape[1]
        for i in range(n_ligands):
            pocket_points = flatten(tf.where(labels[:, i] != 0))
            label = np.max(labels[:, i]) - 1
            
            print(f'Ligand: {label}')
            
            pocket_labels = np.zeros(7, dtype=np.float32)
            pocket_labels[label] = 1.0
            npoints = pocket_points.shape[0]
            if npoints < minPockets:
                continue
            sample = pocket_points

            feed_dict = {
                'input_feat' : tf.gather(data_element[0], pocket_points, axis = 0),
                'rho_coords' : tf.gather(data_element[1], pocket_points, axis = 0),
                'theta_coords' : tf.gather(data_element[2], pocket_points, axis = 0),
                'mask' : tf.gather(data_element[3], pocket_points, axis = 0)
            }
            feed_list.append(feed_dict)
            y_list.append(pocket_labels)
        
        i += 1

    compile_and_save(feed_list, y_list, dataset)

print('Finished!')
