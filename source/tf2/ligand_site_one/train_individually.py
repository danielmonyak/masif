# Header variables and parameters.
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


gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

dev = '/GPU:1'
cpu = '/CPU:0'

#############################################
continue_training = False
#############################################

params = masif_opts["ligand"]

dataset_list = {'train' : "training_data_sequenceSplit_30.tfrecord", 'val' : "validation_data_sequenceSplit_30.tfrecord", 'test' : "testing_data_sequenceSplit_30.tfrecord"}
getData = lambda dataset : tf.data.TFRecordDataset(os.path.join(params["tfrecords_dir"], dataset_list[dataset])).map(_parse_function)
train_data = getData('train')
val_data = getData('val')


model = Sequential([
  layers.Dense(1, activation="relu"),
  layers.Flatten(),
  layers.Dense(64, activation="relu"),
  layers.Dense(20, activation='relu'),
  layers.Dense(1)
])

opt = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits = True)

model.compile(optimizer = opt,
  loss = loss_fn,
  metrics=['binary_accuracy']
)

modelDir = 'kerasModel'
ckpPath = os.path.join(modelDir, 'ckp')
ckpStatePath = ckpPath + '.pickle'

num_epochs = 5

if continue_training:
    model.load_weights(ckpPath)
    with open(ckpStatePath, 'rb') as handle:
        ckpState = pickle.load(handle)
    last_epoch = ckpState['last_epoch']
    best_acc = ckpState['best_acc']
    
    print(f'Loaded model from {ckpStatePath}\nLast completed epoch: {last_epoch}\nValidation accuracy: {best_acc}')
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

batch_threshold = 5e4

with tf.device(dev):
    for i in range(last_epoch + 1, num_epochs):
        
        print(f'Running training data, epoch {i}')
        
        batch_size = 0
        X_list = []
        y_list = []
        for j, data_element in enumerate(train_data):
            if j % 10 == 0:
                print(f'Train record {j}')
                print(f'Current batch size: {batch_size}')
                
            labels = data_element[4]
            if not goodLabel(labels):
                continue

            y_temp = tf.cast(labels > 0, dtype=tf.int32)
            X_temp = tf.constant(data_element[0])

            X_list.append(X_temp)
            y_list.append(y_temp)

            batch_size += len(X_temp)
            if batch_size > batch_threshold:
                print('a')
                X = tf.concat(X_list, axis=0)
                y = tf.concat(y_list, axis=0)
                _=model.fit(X, y, epochs = 1, verbose = 2, class_weight = {0 : 1.0, 1 : 10.0})
                batch_size = 0
                X_list = []
                y_list = []

        if len(X_list) > 0:
            print('b')
            X = tf.concat(X_list, axis=0)
            y = tf.concat(y_list, axis=0)
            _=model.fit(X, y, epochs = 1, verbose = 2, class_weight = {0 : 1.0, 1 : 10.0})
        
        #############################################################
        ###################    VALIDATION DATA    ###################         
        #############################################################
        
        print(f'Running validation data, epoch {i}')
        acc_list = []
        loss_list = []
        
        batch_size = 0
        X_list = []
        y_list = []
        for j, data_element in enumerate(val_data):
            if j % 10 == 0:
                print(f'Validation record {j}')
                print(f'Current batch size: {batch_size}')

            labels = data_element[4]
            if not goodLabel(labels):
                continue

            y_temp = tf.cast(labels > 0, dtype=tf.int32)
            X_temp = tf.constant(data_element[0])

            X_list.append(X_temp)
            y_list.append(y_temp)
            
            batch_size += len(X_temp)
            if batch_size > batch_threshold:
                X = tf.concat(X_list, axis=0)
                y = tf.concat(y_list, axis=0)
                loss, acc = model.evaluate(X, y, verbose = 0)            
                loss_list.append(loss)
                acc_list.append(acc)
                
                batch_size = 0
                X_list = []
                y_list = []
            
        if len(X_list) > 0:
            print('b')
            X = tf.concat(X_list, axis=0)
            y = tf.concat(y_list, axis=0)
            loss, acc = model.evaluate(X, y, verbose = 0)            
            loss_list.append(loss)
            acc_list.append(acc)
            
        
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

print(f'Finished {num_epochs} training epochs!')
