# Header variables and parameters.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import numpy as np
import sys
from default_config.util import *
from tf2.read_ligand_tfrecords import _parse_function
import tensorflow as tf

params = masif_opts["ligand"]
defaultCode = params['defaultCode']
minPockets = params['minPockets']

outdir = '/data02/daniel/masif/datasets/tf2/masif_ligand'
if not os.path.exists(outdir):
    os.mkdir(outdir)
genOutPath = os.path.join(outdir, '{}_{}.npy')

def helper(feed_dict):
    flat_list = list(map(lambda tsr_key : flatten(feed_dict[tsr_key]), data_order))
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
    optional = temp_iterator.get_next_as_optional()
    while optional.has_value():
        data_element = optional.get_value()
        
        print('{} record {}'.format(dataset, i))
        
        labels = data_element[4]
        n_ligands = labels.shape[1]
        for j in range(n_ligands):
            pocket_points = flatten(tf.where(labels[:, j] != 0))
            label = np.max(labels[:, j]) - 1
            
            print(f'Ligand: {label}')
            
            pocket_labels = np.zeros(7, dtype=np.float32)
            pocket_labels[label] = 1.0
            npoints = pocket_points.shape[0]
            if npoints < minPockets:
                continue

            feed_dict = {
                'input_feat' : tf.gather(data_element[0], pocket_points, axis = 0),
                'rho_coords' : tf.gather(data_element[1], pocket_points, axis = 0),
                'theta_coords' : tf.gather(data_element[2], pocket_points, axis = 0),
                'mask' : tf.gather(data_element[3], pocket_points, axis = 0)
            }
            feed_list.append(feed_dict)
            y_list.append(pocket_labels)
        
        i += 1
        optional = temp_iterator.get_next_as_optional()
        
    compile_and_save(feed_list, y_list, dataset)

print('Finished!')
