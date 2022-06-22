# Header variables and parameters.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 


import numpy as np
from IPython.core.debugger import set_trace
import importlib
import sys
from default_config.masif_opts import masif_opts
from read_ligand_tfrecords import _parse_function
import tensorflow as tf

params = masif_opts["ligand"]
defaultCode = params['defaultCode']
minPockets = params['minPockets']

outdir = '/data02/daniel/masif/datasets/tf2/new'
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
    y = tf.stack(y_list, axis = 0)
    np.save(genOutPath.format(dataset, 'X'), X)
    np.save(genOutPath.format(dataset, 'y'), y)

dataset_list = {'train' : "training_data_sequenceSplit_30.tfrecord", 'val' : "validation_data_sequenceSplit_30.tfrecord", 'test' : "testing_data_sequenceSplit_30.tfrecord"}
#dataset_list = {'train' : training_data, 'val' : validation_data, 'test' : testing_data}

for dataset in dataset_list.keys():
    i = 0
    
    feed_list = []
    y_list = []

    temp_data = tf.data.TFRecordDataset(os.path.join(params["tfrecords_dir"], dataset_list[dataset])).map(_parse_function)
    for data_element in temp_data:
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
        sample = pocket_points
        
        feed_dict = {
            'input_feat' : tf.gather(data_element[0], sample, axis = 0),
            'rho_coords' : np.expand_dims(data_element[1], -1)[
                sample, :, :
            ],
            'theta_coords' : np.expand_dims(data_element[2], -1)[
                sample, :, :
            ],
            'mask' : tf.gather(data_element[3], sample, axis = 0)
        }
        feed_list.append(feed_dict)
        y_list.append(pocket_labels)
        
        i += 1

    compile_and_save(feed_list, y_list, dataset)

print('Finished!')
