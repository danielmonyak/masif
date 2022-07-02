import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import sys
import importlib
from IPython.core.debugger import set_trace
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model
from default_config.util import *
from tf2.read_ligand_tfrecords import _parse_function
from MaSIF_ligand_site_one import MaSIF_ligand_site

from time import process_time

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

dev = '/GPU:1'
cpu = '/CPU:0'

#############################################
continue_training = True
read_metrics = False

lastSample = 590
#############################################

params = masif_opts["ligand"]

dataset_list = {'train' : "training_data_sequenceSplit_30.tfrecord", 'val' : "validation_data_sequenceSplit_30.tfrecord", 'test' : "testing_data_sequenceSplit_30.tfrecord"}
getData = lambda dataset : tf.data.TFRecordDataset(os.path.join(params["tfrecords_dir"], dataset_list[dataset])).map(_parse_function)
train_data = getData('train')
val_data = getData('val')


model = MaSIF_ligand_site(
    params["max_distance"],
    params["n_classes"],
    feat_mask=params["feat_mask"]
)
model.compile(optimizer = model.opt,
  loss = model.loss_fn,
  metrics=['binary_accuracy']
)

modelDir = 'kerasModel'
ckpPath = os.path.join(modelDir, 'ckp')
ckpStatePath = ckpPath + '.pickle'

num_epochs = 5
train_samples_threshold = 2e5
val_samples_threshold = 5e4

if continue_training:
    model.load_weights(ckpPath)
    print(f'Loaded model from {ckpPath}')

if read_metrics:
    with open(ckpStatePath, 'rb') as handle:
        ckpState = pickle.load(handle)
    last_epoch = ckpState['last_epoch']
    best_acc = ckpState['best_acc']
    print(f'Last completed epoch: {last_epoch}\nValidation accuracy: {best_acc}')
else:
    last_epoch = -1
    best_acc = 0

def goodLabel(labels):
    n_ligands = labels.shape[1]
    if n_ligands > 1:
        return False
    
    pocket_points = tf.where(labels != 0)
    npoints = tf.shape(pocket_points)[0]
    if npoints < minPockets:
        return False
    
    return True

with tf.device(dev):
    for i in range(last_epoch + 1, num_epochs):
        print(f'Running training data, epoch {i}')
        
        finished_samples = 0
        for j, data_element in enumerate(train_data):
            if j < lastSample:
                continue
            
            print(f'Train record {j}')
            
            labels = data_element[4]
            if not goodLabel(labels):
                continue

            y = tf.cast(labels > 0, dtype=tf.int32)
            X = tuple(data_element[:4])
            _=model.fit(X, y, epochs = 1, verbose = 2, class_weight = {0 : 1.0, 1 : 5.0})
            
            finished_samples += y.shape[0]
            
            if finished_samples > train_samples_threshold:
                finished_samples = 0
                #############################################################
                ###################    VALIDATION DATA    ###################         
                #############################################################
        
                print(f'Testing model on validation data after training on {finished_samples} samples')
                acc_list = []
                loss_list = []

                for j, data_element in enumerate(val_data):
                    print(f'Validation record {j}')

                    labels = data_element[4]
                    if not goodLabel(labels):
                        continue

                    y = tf.cast(labels > 0, dtype=tf.int32)
                    X = tuple(data_element[:4])

                    loss, acc = model.evaluate(X, y, verbose = 0)            
                    loss_list.append(loss)
                    acc_list.append(acc)

                    finished_samples += y.shape[0]
                    
                    if finished_samples > val_samples_threshold:
                        finished_samples = 0
                        acc = sum(acc_list)/len(acc_list)
                        loss = sum(loss_list)/len(loss_list)
                        print(f'Epoch {i} finished\nLoss: {round(loss, 2)}\nBinary Accuracy: {round(acc, 2)}')

                        if acc > best_acc:
                            print(f'Validation accuracy improved from {best_acc} to {acc}')
                            print(f'Saving model weights to {ckpPath}')
                            best_acc = acc
                            model.save_weights(ckpPath)
                            ckpState = {'best_acc' : best_acc, 'last_epoch' : i}
                            with open(ckpStatePath, 'wb') as handle:
                                pickle.dump(ckpState, handle, protocol=pickle.HIGHEST_PROTOCOL)
