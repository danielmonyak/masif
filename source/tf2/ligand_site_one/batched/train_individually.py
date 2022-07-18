import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import sys
import importlib
from IPython.core.debugger import set_trace
import pickle
import numpy as np
import tensorflow as tf
'''
phys_gpus = tf.config.list_physical_devices('GPU')
for phys_g in phys_gpus:
    tf.config.experimental.set_memory_growth(phys_g, True)
'''
from default_config.util import *
from tf2.read_ligand_tfrecords import _parse_function
from MaSIF_ligand_site_one import MaSIF_ligand_site

dev = '/GPU:1'
cpu = '/CPU:0'

log_gpus = tf.config.list_logical_devices('GPU')
gpu_strs = [g.name for g in log_gpus]
strategy = tf.distribute.MirroredStrategy(gpu_strs)

tf.config.set_soft_device_placement(True)
#tf.debugging.set_log_device_placement(True)

#############################################
continue_training = False
#read_metrics = False

starting_sample = 0
starting_epoch = 0
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

cv_batch_sz = 50

#with tf.device(dev):
with strategy.scope():
    model = MaSIF_ligand_site(
        params["max_distance"],
        params["n_classes"],
        feat_mask=params["feat_mask"],
        #n_conv_layers = masif_opts['site']['n_conv_layers'],
        n_conv_layers = 2,
        conv_batch_size = cv_batch_sz
    )

    from_logits = model.loss_fn.get_config()['from_logits']
    binAcc = tf.keras.metrics.BinaryAccuracy(threshold = (not from_logits) * 0.5)
    auc = tf.keras.metrics.AUC(from_logits = from_logits)

    model.compile(optimizer = model.opt,
      loss = model.loss_fn,
      metrics=[binAcc, auc]
    )

    if continue_training:
        model.load_weights(ckpPath)
        print(f'Loaded model from {ckpPath}')
    '''
    if read_metrics:
        with open(ckpStatePath, 'rb') as handle:
            ckpState = pickle.load(handle)
        starting_epoch = ckpState['last_epoch']
        print(f'Resuming epoch {i} of training\nValidation accuracy: {best_acc}')
    else:
        i = 0
        best_acc = 0
    '''



    #######################################
    #######################################
    #######################################

    i = starting_epoch

    print(f'Running training data, epoch {i}')
    while i < num_epochs:
        train_iterator = iter(train_data)
        train_j = 0
        if i == starting_epoch:
            while train_j < starting_sample:
                data_element = train_iterator.get_next()
                train_j += 1
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
            y = (y_added > 0).astype(np.int32)

            n_samples = y.shape[0]

            ##### 200 every time
            max_verts = data_element[0].shape[1]
            #####

            pdb = data_element[5].numpy().decode('ascii') + '_'
            indices = np.load(os.path.join(params['masif_precomputation_dir'], pdb, 'p1_list_indices.npy'), encoding="latin1", allow_pickle = True)
            indices = pad_indices(indices, max_verts).astype(np.int32)

            data_tsrs = tuple(np.expand_dims(tsr, axis=0) for tsr in data_element[:4])
            indices = np.expand_dims(indices, axis=0)
            y = np.expand_dims(y, axis=0)

            X = (data_tsrs, indices)

            model.fit(X, y, verbose = 2)

            print('\n\nFinished training on one protein\n\n')
            finished_samples += n_samples

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
