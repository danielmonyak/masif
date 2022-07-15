import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import sys
import pickle
import numpy as np
import tensorflow as tf
from default_config.util import *
from tf2.read_ligand_tfrecords import _parse_function
from tf2.LSResNet.LSResNet import LSResNet
from tf2.usage.predictor import Predictor
import tfbio.data

gpus_IN_USE = [2,3]

phys_gpus = tf.config.list_physical_devices('GPU')
for i in gpus_IN_USE:
    tf.config.experimental.set_memory_growth(phys_gpus[i], True)

lg_gpus = tf.config.experimental.list_logical_devices('GPU')
gpus_str = [g.name for g in lg_gpus]
strategy = tf.distribute.MirroredStrategy([gpus_str[i] for i in gpus_IN_USE])

dev = '/GPU:1'
cpu = '/CPU:0'

precom_dir = '/data02/daniel/masif/masif_ligand/data_preparation/04a-precomputation_12A/precomputation'

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

def goodLabel(labels):
    n_ligands = labels.shape[1]
    if n_ligands > 1:
        return False
    
    pocket_points = tf.where(labels != 0)
    npoints = tf.shape(pocket_points)[0]
    if npoints < minPockets:
        return False
    
    return True

#with tf.device(dev):
with strategy.scope():
    model = LSResNet(
        params["max_distance"],
        params["n_classes"],
        feat_mask=params["feat_mask"]
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
        
        optional = temp_iterator.get_next_as_optional()
        while optional.has_value():
            data_element = optional.get_value()
                
            print(f'Epoch {i}, train record {train_j}')

            labels = data_element[4]
            if not goodLabel(labels):
                train_j += 1
                continue

            pdb = data_element[5].numpy().decode('ascii') + '_'
            pdb_dir = os.path.join(precom_dir, pdb)
            xyz_coords = tf.cast(tf.expand_dims(Predictor.getXYZCoords(pdb_dir), axis=0), dtype=tf.float32)
            
            '''
            #X = tuple(tf.expand_dims(tsr, axis=0) for tsr in data_element[:4])
            coords = [tf.expand_dims(tsr, axis=-1) for tsr in data_element[1:3]]
            X = tf.expand_dims(tf.concat([data_element[0]] + coords + [data_element[3]], axis=-1), axis=0)
            '''
            X = data_element[:4]
            
            y_raw = tf.cast(labels > 0, dtype=tf.int32)
            resolution = 1. / model.scale
            y = tfbio.data.make_grid(xyz_coords[0], y_raw, max_dist=model.max_dist, grid_resolution=resolution)
            
            X_packed = (X, xyz_coords)
            
            _=model.fit(X_packed, y, verbose = 2, use_multiprocessing = True)
            
            finished_samples += labels.shape[0]
            train_j += 1
            
            optional = temp_iterator.get_next_as_optional()
            
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
