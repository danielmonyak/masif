# Header variables and parameters.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 


import numpy as np
from IPython.core.debugger import set_trace
import importlib
import sys
from default_config.masif_opts import masif_opts
#####
# Edited by Daniel Monyak
from MaSIF_ligand_site import MaSIF_ligand_site
#####
from tf2.read_ligand_tfrecords import _parse_function
import tensorflow as tf
#import time

params = masif_opts["ligand"]
defaultCode = params['defaultCode']
n_classes = params['n_classes']

# Load dataset
training_data = tf.data.TFRecordDataset(
    os.path.join(params["tfrecords_dir"], "training_data_sequenceSplit_30.tfrecord")
)
validation_data = tf.data.TFRecordDataset(
    os.path.join(params["tfrecords_dir"], "validation_data_sequenceSplit_30.tfrecord")
)
testing_data = tf.data.TFRecordDataset(
    os.path.join(params["tfrecords_dir"], "testing_data_sequenceSplit_30.tfrecord")
)
training_data = training_data.map(_parse_function)
validation_data = validation_data.map(_parse_function)
testing_data = testing_data.map(_parse_function)

outdir = 'datasets/'

dataset_list = {'train' : training_data, 'val' : validation_data, 'test' : testing_data}
for dataset in dataset_list.keys():
    print('\n' + dataset)
    i = 0
    
    feed_list = []
    y_list = []
    for data_element in dataset_list[dataset]:
        print('{} record {}'.format(dataset, i))
        
        labels = data_element[4]
        n_ligands = labels.shape[1]
        if n_ligands > 1:
            print('More than one ligand, check this out...')
        
        one_hot_labels = tf.one_hot(tf.reshape(labels, [-1,]) - 1, n_classes)
        
        feed_dict = {
            'input_feat' : data_element[0],
            'rho_coords' : np.expand_dims(data_element[1], -1),
            'theta_coords' : np.expand_dims(data_element[2], -1),
            'mask' : data_element[3],
        }
        feed_list.append(feed_dict)
        y_list.append(one_hot_labels)
        
        i += 1

    tsr_list = []
    for feed_dict in feed_list:
        flat_list = []
        for tsr_key in ['input_feat', 'rho_coords', 'theta_coords', 'mask']:
            tsr = feed_dict[tsr_key]
            flat_list.append(tf.reshape(tsr, [-1]))
        tsr_list.append(tf.concat(flat_list, axis = 0))
    
    X = tf.ragged.stack(tsr_list).to_tensor(default_value = defaultCode)
    y = tf.stack(y_list, axis = 0)

    np.save(outdir + '{}_X.npy'.format(dataset), X)
    np.save(outdir + '{}_y.npy'.format(dataset), y)

print('Finished!')
