import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import sys
import importlib
from IPython.core.debugger import set_trace
import pickle
import numpy as np
import tensorflow as tf


phys_gpus = tf.config.list_physical_devices('GPU')
for phys_g in phys_gpus:
    tf.config.experimental.set_memory_growth(phys_g, True)
'''
lg_gpus = tf.config.experimental.list_logical_devices('GPU')
gpus_str = [g.name for g in lg_gpus]
strategy = tf.distribute.MirroredStrategy([gpus_str[i] for i in gpus_IN_USE])
'''
from default_config.util import *
from tf2.read_ligand_tfrecords import _parse_function
from tf2.ligand_site_one.MaSIF_ligand_site_one import MaSIF_ligand_site

dev = '/GPU:1'
cpu = '/CPU:0'

#############################################
continue_training = False
read_metrics = False

starting_sample = 0
#############################################

params = masif_opts["ligand"]

dataset_list = {'train' : "training_data_sequenceSplit_30.tfrecord", 'val' : "validation_data_sequenceSplit_30.tfrecord", 'test' : "testing_data_sequenceSplit_30.tfrecord"}
getData = lambda dataset : tf.data.TFRecordDataset(os.path.join(params["tfrecords_dir"], dataset_list[dataset])).map(_parse_function)
train_data = getData('train')

modelDir = 'kerasModel'
ckpPath = os.path.join(modelDir, 'ckp')
ckpStatePath = ckpPath + '.pickle'

modelPath_endTraining = os.path.join(modelDir, 'savedModel_endTraining')

#############################################
#############################################
num_epochs = 5                  #############
pdb_ckp_thresh = 10             #############
#############################################
#############################################

max_verts = 200

with tf.device(dev):
#with strategy.scope():
    model = MaSIF_ligand_site(
        params["max_distance"],
        params["n_classes"],
        feat_mask=params["feat_mask"],
        n_conv_layers = 3,
        conv_batch_size = 500
    )

    from_logits = model.loss_fn.get_config()['from_logits']
    binAcc = tf.keras.metrics.BinaryAccuracy(threshold = (not from_logits) * 0.5)

    model.compile(optimizer = model.opt,
      loss = model.loss_fn,
      metrics=[binAcc]
    )
    
    if continue_training:
        model.load_weights(ckpPath)
        print(f'Loaded model from {ckpPath}')

    if read_metrics:
        with open(ckpStatePath, 'rb') as handle:
            ckpState = pickle.load(handle)
        i = ckpState['last_epoch']
        best_acc = ckpState['best_acc']
        print(f'Resuming epoch {i} of training\nValidation accuracy: {best_acc}')
    else:
        i = 0
        best_acc = 0

    #######################################
    #######################################
    #######################################
    
    train_iterator = iter(train_data)

    train_j = 0
    val_j = 0
    while train_j < starting_sample:
        data_element = train_iterator.get_next()
        train_j += 1
    
    print(f'Running training data, epoch {i}')
    while i < num_epochs:
        #############################################################
        ###################     TRAINING DATA     ###################
        #############################################################
        finished_samples = 0
        
        optional = train_iterator.get_next_as_optional()
        while optional.has_value():
            data_element = optional.get_value()
            
            print(f'Epoch {i}, train record {train_j}')
            
            labels = data_element[4]
            bin_labels = np.asarray(labels > 0).astype(int)
            pocket_points_count = np.sum(bin_labels, axis=0)
            good_labels = bin_labels[:, pocket_points_count > minPockets]
            if good_labels.shape[1] == 0:
                train_j += 1
                optional = train_iterator.get_next_as_optional()
                continue
            
            y_added = np.sum(good_labels, axis=1, keepdims=True)
            
            y = tf.cast(y_added > 0, dtype=tf.int32)
            batch_sz = y.shape[0]
            
            pdb = data_element[5].numpy().decode('ascii') + '_'
            indices = np.load(os.path.join(params['masif_precomputation_dir'], pdb, 'p1_list_indices.npy'), encoding="latin1", allow_pickle = True)
            indices = pad_indices(indices, max_verts)
            
            X = (data_element[:4], indices)

            
            #class_weight = {0 : 1.0, 1 : 20.0})
            model.fit(X, y, verbose = 2)
            
            print('\n\nFinished training on one protein\n\n')
            finished_samples += batch_sz
            
            train_j += 1
            optional = train_iterator.get_next_as_optional()
            
            if train_j % pdb_ckp_thresh == 0:
                print(f'Saving model weights to {ckpPath}')
                model.save_weights(ckpPath)
        
        train_iterator = iter(train_data)
        train_j = 0
        i += 1
        
        print(f'Saving model weights to {ckpPath}')
        model.save_weights(ckpPath)

print(f'Saving model to to {modelPath_endTraining}')
model.save(modelPath_endTraining)
