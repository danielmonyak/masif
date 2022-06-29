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

from operator import add
from time import process_time

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

dev = '/GPU:1'
cpu = '/CPU:0'

#############################################
continue_training = True
#############################################

params = masif_opts["ligand"]
minPockets = params['minPockets']

dataset_list = {'train' : "training_data_sequenceSplit_30.tfrecord", 'val' : "validation_data_sequenceSplit_30.tfrecord", 'test' : "testing_data_sequenceSplit_30.tfrecord"}
getData = lambda dataset : tf.data.TFRecordDataset(os.path.join(params["tfrecords_dir"], dataset_list[dataset])).map(_parse_function)
train_data = getData('train')
val_data = getData('val')

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
ckpStatePath = ckpPath + '.pickle'

num_epochs = 100

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
        #print('More than one ligand, check this out...')
        return False
    
    pocket_points = tf.where(labels != 0)
    npoints = tf.shape(pocket_points)[0]
    if npoints < minPockets:
        #print(f'Only {npoints} pocket_points')
        return False
    
    return True

with tf.device(dev):
    for i in range(last_epoch + 1, num_epochs):
        print(f'Running training data, epoch {i}')
        
        #batch_size = 0
        for j, data_element in enumerate(train_data):
            if j == 5:
                break
            
            if j % 100 == 0:
                print(f'Train record {j}')
                verbosity = 2
                #print('Time:', round(process_time(), 3))
            
            labels = data_element[4]
            if not goodLabel(labels):
                continue
            
            y_raw = tf.transpose(tf.cast(labels > 0, dtype=tf.int32))
            
            #flat_list = list(map(flatten, data_element[:4]))
            #X = tf.expand_dims(tf.concat(flat_list, axis=0), axis=0)
            #_=model.fit(X, y_raw, epochs = 1, verbose = 0)
            
            #####
            n_pockets = y_raw.shape[1]
            fullSamples = n_pockets // minPockets
            for i in range(fullSamples):
                if i % 10 == 0:
                    done = 100.0 * i/fullSamples
                    print('{} of {} batches completed'.format(i, fullSamples))
                sample = range(minPockets * i, minPockets * (i+1))
                
                y_temp = tf.gather(y_raw, sample, axis=1)
                flat_list = list(map(lambda tsr : np.take(tsr, sample, axis=0).flatten(), data_element[:4]))
                X = tf.expand_dims(tf.concat(flat_list, axis=0), axis=0)
                
                _=model.fit(X, y_temp, epochs = 1, verbose = 0)
            
            i = fullSamples
            n_leftover = n_pockets % minPockets
            garbage = tf.range(minPockets * (i-1) + n_leftover, minPockets * i)
            valid = tf.range(minPockets * i, minPockets * i + n_leftover)
            sample = tf.concat([garbage, valid], axis=0)
            
            y_temp = tf.gather(y_raw, sample, axis=1)
            flat_list = list(map(lambda tsr : np.take(tsr, sample, axis=0).flatten(), data_element[:4]))
            X = tf.expand_dims(tf.concat(flat_list, axis=0), axis=0)

            _=model.fit(X, y_temp, epochs = 1, verbose = verbosity)
            verbosity = 0
            #####
            
            
            #batch_size += 1
            '''
            y, sample = model.make_y(y_raw)
            with tf.GradientTape() as tape:
                y_pred = model(X, sample = sample, training=True)
                loss = model.compiled_loss(y, y_pred, regularization_losses=model.losses)
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            '''
            '''if batch_size == 1:
                tempGradients = gradients.copy()
            else:
                tempGradients = list(map(add, tempGradients, gradients))'''
        
        #gradients = [grad / batch_size for grad in tempGradients]
        #model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        print(f'Running validation data, epoch {i}')
        acc_list = []
        loss_list = []
        for j, data_element in enumerate(val_data):
            if j == 5:
                break
            
            if j % 10 == 0:
                print(f'Validation record {j}')

            labels = data_element[4]
            if not goodLabel(labels):
                continue

            y_raw = tf.transpose(tf.cast(labels > 0, dtype=tf.int32))

            #flat_list = list(map(flatten, data_element[:4]))
            #X = tf.expand_dims(tf.concat(flat_list, axis=0), axis=0)
            #loss, acc = model.evaluate(X, y_raw, verbose = 0)            
            #loss_list.append(loss)
            #acc_list.append(acc)
        
            #####
            n_pockets = y_raw.shape[1]
            fullSamples = n_pockets // minPockets
            for i in range(fullSamples):
                sample = range(minPockets * i, minPockets * (i+1))
                
                y_temp = tf.gather(y_raw, sample, axis=1)
                flat_list = list(map(lambda tsr : np.take(tsr, sample, axis=0).flatten(), data_element[:4]))
                X = tf.expand_dims(tf.concat(flat_list, axis=0), axis=0)
                
                loss, acc = model.evaluate(X, y_temp, verbose = 0)            
                loss_list.append(loss)
                acc_list.append(acc)
            
            i = fullSamples
            n_leftover = n_pockets % minPockets
            garbage = tf.range(minPockets * (i-1) + n_leftover, minPockets * i)
            valid = tf.range(minPockets * i, minPockets * i + n_leftover)
            sample = tf.concat([garbage, valid], axis=0)
            
            y_temp = tf.gather(y_raw, sample, axis=1)
            flat_list = list(map(lambda tsr : np.take(tsr, sample, axis=0).flatten(), data_element[:4]))
            X = tf.expand_dims(tf.concat(flat_list, axis=0), axis=0)

            loss, acc = model.evaluate(X, y_temp, verbose = 0)            
            loss_list.append(loss)
            acc_list.append(acc)
            #####
        
        acc = sum(acc_list)/len(acc_list)
        loss = sum(loss_list)/len(loss_list)
        print(f'Epoch {i} finished\nLoss: {loss}\nBinary Accuracy: {acc}')
        
        if acc > best_acc:
            print(f'Validation accuracy improved from {best_acc} to {acc}')
            print(f'Saving model weights to {ckpPath}')
            best_acc = acc
            model.save_weights(ckpPath)
            ckpState = {'best_acc' : best_acc, 'last_epoch' : i}
            with open(ckpStatePath, 'wb') as handle:
                pickle.dump(ckpState, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(f'Finished {num_epochs} training epochs!')
