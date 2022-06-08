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
from MaSIF_ligand_TF2 import MaSIF_ligand
#####
from read_ligand_tfrecords import _parse_function
import tensorflow as tf
#import time

params = masif_opts["ligand"]
defaultCode = params['defaultCode']

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

minPockets = params['minPockets']

outdir = 'datasets/'

dataset_list = {'train' : training_data, 'val' : validation_data, 'test' : testing_data}
for dataset in dataset_list.keys():
    print('\n' + dataset)
    i = 0
    
    feed_list = []
    y_list = []
    for data_element in dataset_list[dataset]:
        print('{} record {}'.format(dataset, i))
        
        random_ligand = 0
        labels = data_element[4]
        n_ligands = labels.shape[1]
        pocket_points = tf.reshape(tf.where(labels[:, random_ligand] != 0), [-1, ])
        label = np.max(labels[:, random_ligand]) - 1
        pocket_labels = np.zeros(7, dtype=np.float32)
        pocket_labels[label] = 1.0
        npoints = pocket_points.shape[0]
        if npoints < minPockets:
            continue
        # select random pockets in training, not prep_data.py
        #sample = np.random.choice(pocket_points, minPockets, replace=False)
        sample = pocket_points
        
        feed_dict = {
            'input_feat' : tf.gather(data_element[0], sample, axis = 0),
            'rho_coords' : np.expand_dims(data_element[1], -1)[
                sample, :, :
            ],
            'theta_coords' : np.expand_dims(data_element[2], -1)[
                sample, :, :
            ],
            'mask' : tf.gather(data_element[3], pocket_points[:minPockets], axis = 0)
        }
        feed_list.append(feed_dict)
        y_list.append(pocket_labels)
        
        i += 1

    tsr_list = []
    for feed_dict in feed_list:
        flat_list = []
        for tsr_key in ['input_feat', 'rho_coords', 'theta_coords', 'mask']:
            tsr = feed_dict[tsr_key]
            flat_list.append(tf.reshape(tsr, [-1]))
        tsr_list.append(tf.concat(flat_list, axis = 0))
    
    #X = tf.stack(tsr_list)
    X = tf.ragged.stack(tsr_list).to_tensor(default_value = defaultCode)
    y = tf.stack(y_list, axis = 0)

    np.save(outdir + '{}_X.npy'.format(dataset), X)
    np.save(outdir + '{}_y.npy'.format(dataset), y)

print('Finished!')
