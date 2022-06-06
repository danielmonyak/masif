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

training_data = iter(training_data)
validation_data = iter(validation_data)
testing_data = iter(testing_data)



# Create Model
model = MaSIF_ligand(
  params["max_distance"],
  params["n_classes"],
  idx_gpu="/gpu:0",
  feat_mask=params["feat_mask"],
  costfun=params["costfun"]
)

model.compile(optimizer = model.opt,
  loss = model.loss_fn,
  metrics=['accuracy']
)
'''
i = 0
for data_element in training_data:
    random_ligand = 0
    labels = data_element[4]
    n_ligands = labels.shape[1]
    #pocket_points = np.where(labels[:, random_ligand] != 0)[0]
    pocket_points = tf.reshape(tf.where(labels[:, random_ligand] != 0), [-1, ])
    label = np.max(labels[:, random_ligand]) - 1
    pocket_labels = np.zeros(7, dtype=np.float32)
    pocket_labels[label] = 1.0
    npoints = pocket_points.shape[0]
    if npoints < 32:
        continue
    # Sample 32 points randomly
    # Fix later - otherwise it's same random points every epoch
    sample = np.random.choice(pocket_points, 32, replace=False)
    feed_dict = {
        'input_feat' : data_element[0][sample, :, :],
        'rho_coords' : np.expand_dims(data_element[1], -1)[
            sample, :, :
        ],
        'theta_coords' : np.expand_dims(data_element[2], -1)[
            sample, :, :
        ],
        'mask' : data_element[3][pocket_points[:32], :, :],
        'labels' : pocket_labels,
        'keep_prob' : 1.0,
    }
    output = 
'''
