# Header variables and parameters.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import sys
import importlib
from IPython.core.debugger import set_trace
import pickle
import numpy as np
import tensorflow as tf
from default_config.util import *
from tf2.read_ligand_tfrecords import _parse_function
from tf2.ligand_site.MaSIF_ligand_site import MaSIF_ligand_site

continue_training = False

params = masif_opts["ligand"]
minPockets = params['minPockets']
dataset_list = {'train' : "training_data_sequenceSplit_30.tfrecord", 'val' : "validation_data_sequenceSplit_30.tfrecord", 'test' : "testing_data_sequenceSplit_30.tfrecord"}
train_data = tf.data.TFRecordDataset(os.path.join(params["tfrecords_dir"], dataset_list['train'])).map(_parse_function)
val_data = tf.data.TFRecordDataset(os.path.join(params["tfrecords_dir"], dataset_list['val'])).map(_parse_function)


model = MaSIF_ligand_site(
    params["max_distance"],
    params["n_classes"],
    feat_mask=params["feat_mask"],
    n_conv_layers = 1
)
model.compile(optimizer = model.opt,
    loss = model.loss_fn,
    metrics=['binary_accuracy']
)
modelDir = 'kerasModel'
ckpPath = os.path.join(modelDir, 'ckp')

if continue_training:
    model.load_weights(ckpPath)
    last_epoch += 13
    initValThresh = 0.91657

num_epochs = 100

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

dev = '/GPU:1'
cpu = '/CPU:0'

def goodLabel(labels):
    n_ligands = labels.shape[1]
    if n_ligands > 1:
        print('More than one ligand, check this out...')
        return False

    #labels = tf.squeeze(labels)
    pocket_points = tf.squeeze(tf.where(labels != 0))
    npoints = tf.shape(pocket_points)[0]
    if npoints < minPockets:
        print('Only {} pocket_points'.format(npoints))
        return False
    return True

for i in epochs:
    with tf.device(dev):
        print('Running training data, epoch {}'.format(i))
        for j, data_element in enumerate(train_data):
            print('Train record {}'.format(j))

            labels = data_element[4]
            if not goodLabel(labels):
                print('Skipping this record...')
                continue
            
            feed_dict = {
                'input_feat' : data_element[0],
                'rho_coords' : tf.expand_dims(data_element[1], -1),
                'theta_coords' : tf.expand_dims(data_element[2], -1),
                'mask' : data_element[3]
            }
            flat_list = list(map(lambda tsr_key : flatten(feed_dict[tsr_key]), data_order))
            
            X = tf.concat(flat_list, axis=0)
            model.fit(X, labels, epochs = 1, verbose = 2)
        
        print('Running validation data, epoch {}'.format(i))
        for j, data_element in enumerate(val_data):
            print('Validation record {}'.format(j))
            
            labels = data_element[4]
            if not goodLabel(labels):
                print('Skipping this record...')
                continue
            
            feed_dict = {
                'input_feat' : data_element[0],
                'rho_coords' : tf.expand_dims(data_element[1], -1),
                'theta_coords' : tf.expand_dims(data_element[2], -1),
                'mask' : data_element[3]
            }
            flat_list = list(map(lambda tsr_key : flatten(feed_dict[tsr_key]), data_order))
            
            X = tf.concat(flat_list, axis=0)
            model.evaluate(X, labels, verbose = 2)
                
