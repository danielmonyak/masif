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
from sklearn.metrics import confusion_matrix
import tensorflow as tf

continue_training = True


params = masif_opts["ligand"]


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
'''
@tf.function
def make_iter(data):
    return iter(data)
training_data = make_iter(training_data)
validation_data = make_iter(validation_data)
testing_data = make_iter(testing_data)
'''

modelDir = 'kerasModel'

# Create Model
model = MaSIF_ligand(
  params["max_distance"],
  params["n_classes"],
  feat_mask=params["feat_mask"]
)



i = 0
for data_element in training_data:
    random_ligand = 0
    labels = data_element[4]
    n_ligands = labels.shape[1]
    pocket_points = tf.reshape(tf.where(labels[:, random_ligand] != 0), [-1, ])
    label = np.max(labels[:, random_ligand]) - 1
    pocket_labels = np.zeros(7, dtype=np.float32)
    pocket_labels[label] = 1.0
    npoints = pocket_points.shape[0]
    
    # Sample model.minPockets (32) points randomly
    # Fix later - otherwise it's same random points every epoch
    sample = np.random.choice(pocket_points, model.minPockets, replace=False)
    feed_dict = {
        'input_feat' : tf.gather(data_element[0], sample, axis = 0),
        'rho_coords' : tf.gather(tf.expand_dims(data_element[1], -1), sample, axis = 0),
        'theta_coords' : tf.gather(tf.expand_dims(data_element[2], -1), sample, axis = 0),
        'mask' : tf.gather(data_element[3], pocket_points[:model.minPockets], axis = 0)
    }
    
    if npoints >= model.minPockets:
        break

tsr_list = [tf.transpose(tsr, perm=[0,2,1]) for tsr in feed_dict.values()]
ragged_input = tf.ragged.stack(tsr_list)
'''outdir = 'datasets/'

X_list = []
y_list = []

i = 0
dataset_list = {'train' : training_data, 'val' : validation_data, 'test' : testing_data}
print('start')
for dataset in dataset_list.keys():
    print('\n' + dataset)
    for data_element in dataset_list[dataset]:
        print(i)
        random_ligand = 0
        labels = data_element[4]
        n_ligands = labels.shape[1]
        pocket_points = tf.reshape(tf.where(labels[:, random_ligand] != 0), [-1, ])
        label = np.max(labels[:, random_ligand]) - 1
        pocket_labels = np.zeros(7, dtype=np.float32)
        pocket_labels[label] = 1.0
        npoints = pocket_points.shape[0]
        if npoints < model.minPockets:
            continue
        # Sample model.minPockets (32) points randomly
        # Fix later - otherwise it's same random points every epoch
        sample = np.random.choice(pocket_points, model.minPockets, replace=False)
        feed_dict = {
            'input_feat' : tf.gather(data_element[0], sample, axis = 0),
            'rho_coords' : np.expand_dims(data_element[1], -1)[
                sample, :, :
            ],
            'theta_coords' : np.expand_dims(data_element[2], -1)[
                sample, :, :
            ],
            'mask' : tf.gather(data_element[3], pocket_points[:model.minPockets], axis = 0)
        }
        ret = model.bigPrepData(feed_dict)
        X_list.append(ret)
        y_list.append(pocket_labels)
        i += 1

    X = tf.stack(X_list, axis = 0)
    y = tf.stack(y_list, axis = 0)

    np.save(outdir + '{}_X.npy'.format(dataset), X)
    np.save(outdir + '{}_y.npy'.format(dataset), y)
'''
