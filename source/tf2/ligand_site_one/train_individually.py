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
from tf2.ligand_site_one.MaSIF_ligand_site_one import MaSIF_ligand_site
'''
phys_gpus = tf.config.list_physical_devices('GPU')
strategy_str = []
for phys_g in phys_gpus:
    tf.config.experimental.set_memory_growth(phys_g, True)
'''
lg_gpus = tf.config.experimental.list_logical_devices('GPU')
gpus_str = [g.name for g in lg_gpus]
strategy = tf.distribute.MirroredStrategy(gpus_str)

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
val_data = getData('val')


modelDir = 'kerasModel'
ckpPath = os.path.join(modelDir, 'ckp')
ckpStatePath = ckpPath + '.pickle'

modelPath_endTraining = os.path.join(modelDir, 'savedModel_endTraining')

#############################################
#############################################
num_epochs = 5                  #############
train_samples_threshold = 2e5   #############
val_samples_threshold = 5e4     #############
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

max_verts = 200
'''
def pad_map_fn(packed):
    row, def_val = packed
    leftover = max_verts - row.shape[0]
    paddings = tf.constant([[0,leftover]])
    return tf.pad(row, paddings, constant_values=def_val)'''

def pad_indices(indices, max_verts):
    ret_list = []
    for patch_ix in range(len(indices)):
        ret_list.append(np.concatenate(
            [indices[patch_ix], [patch_ix] * (max_verts - len(indices[patch_ix]))])
        )
    return np.stack(ret_list)


#with tf.device(dev):
with strategy.scope():
    model = MaSIF_ligand_site(
        params["max_distance"],
        params["n_classes"],
        feat_mask=params["feat_mask"],
        n_conv_layers = 3
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
    val_iterator = iter(val_data)

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
        
        while finished_samples < train_samples_threshold:
            optional = train_iterator.get_next_as_optional()
            if optional.has_value():
                data_element = optional.get_value()
            else:
                train_iterator = iter(train_data)
                train_j = 0
                i += 1
                print(f'Running training data, epoch {i}')
                continue
                
            print(f'Epoch {i}, train record {train_j}')

            labels = data_element[4]
            if not goodLabel(labels):
                train_j += 1
                continue

            pdb = data_element[5].numpy().decode('ascii') + '_'
            indices = np.load(os.path.join(params['masif_precomputation_dir'], pdb, 'p1_list_indices.npy'), encoding="latin1", allow_pickle = True)
            
            '''default_values = tf.range(indices.shape[0])
            indices = tf.map_fn(fn=pad_map_fn, elems=(indices, default_values), fn_output_signature=tf.TensorSpec(shape=max_verts, dtype=tf.int32))'''
            indices = np.expand_dims(pad_indices(indices, max_verts), axis=-1)
            
            
            '''pocket_points = np.where(np.squeeze(labels > 0))[0]
            npoints = pocket_points.shape[0]
            empty_points = np.where(np.squeeze(labels == 0))[0]
            #empty_sample = tf.random.shuffle(empty_points)[:npoints]
            empty_sample = np.random.choice(empty_points, npoints)
            
            #sample = flatten(tf.concat([pocket_points, empty_sample], axis=0))
            sample = np.concatenate([pocket_points, empty_sample])'''
            
            #coords = [tf.expand_dims(tsr, axis=-1) for tsr in data_element[1:3]]
            #X = tf.expand_dims(tf.concat([data_element[0]] + coords + [data_element[3], indices], axis=-1), axis=0)
            coords = [np.expand_dims(tsr, axis=-1) for tsr in data_element[1:3]]
            X = np.concatenate([data_element[0]] + coords + [data_element[3], indices], axis=-1)

            y = tf.cast(labels > 0, dtype=tf.int32)
            
            '''y_samp = tf.gather(y, sample)
            X_samp = X[sample]'''
            
            batch_sz = X.shape[0]
            
            #model.fit(X, y, verbose = 1, class_weight = {0 : 1.0, 1 : 20.0})
            model.fit(X, y, verbose = 1, batch_size = batch_sz, use_multiprocessing = True)

            finished_samples += batch_sz
            train_j += 1
            
        #############################################################
        ###################    VALIDATION DATA    ###################
        #############################################################
        print(f'Testing model on validation data after training on {finished_samples} samples')
        
        finished_samples = 0
        
        acc_list = []
        loss_list = []

        while finished_samples < val_samples_threshold:
            optional = val_iterator.get_next_as_optional()
            if optional.has_value():
                data_element = optional.get_value()
            else:
                val_iterator = iter(val_data)
                val_j = 0
                continue
            
            print(f'Validation record {val_j}')

            labels = data_element[4]
            if not goodLabel(labels):
                continue

            pdb = data_element[5].numpy().decode('ascii') + '_'
            indices = np.load(os.path.join(params['masif_precomputation_dir'], pdb, 'p1_list_indices.npy'), encoding="latin1", allow_pickle = True)
            indices = np.expand_dims(pad_indices(indices, max_verts), axis=-1)
            
            '''pocket_points = np.where(np.squeeze(labels > 0))[0]
            npoints = pocket_points.shape[0]
            empty_points = np.where(np.squeeze(labels == 0))[0]
            empty_sample = np.random.choice(empty_points, npoints)
            
            sample = np.concatenate([pocket_points, empty_sample])'''
            
            coords = [np.expand_dims(tsr, axis=-1) for tsr in data_element[1:3]]
            X = np.concatenate([data_element[0]] + coords + [data_element[3], indices], axis=-1)
            y = tf.cast(labels > 0, dtype=tf.int32)
            '''
            y_samp = tf.gather(y, sample)
            X_samp = X[sample]'''
            
            batch_sz = X.shape[0]
            
            loss, acc = model.evaluate(X, y, verbose = 0, batch_size = batch_sz)            
            loss_list.append(loss)
            acc_list.append(acc)

            finished_samples += batch_sz
            val_j += 1
        
        #############################################################
        #############    EVALUATING VALIDATION DATA    ##############
        #############################################################
        
        acc = sum(acc_list)/len(acc_list)
        loss = sum(loss_list)/len(loss_list)
        print(f'Finished evaluating {finished_samples} validation samples\nLoss: {round(loss, 2)}\nBinary Accuracy: {round(acc, 2)}')

        if acc > best_acc:
            print(f'Validation accuracy improved from {best_acc} to {acc}')
            print(f'Saving model weights to {ckpPath}')
            best_acc = acc
            model.save_weights(ckpPath)
            ckpState = {'best_acc' : best_acc, 'last_epoch' : i}
            with open(ckpStatePath, 'wb') as handle:
                pickle.dump(ckpState, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            print(f'Validation accuracy did not improve from {best_acc}')

model.save(modelPath_endTraining)
            
