import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import sys
import numpy as np
import tensorflow as tf

phys_gpus = tf.config.list_physical_devices('GPU')
for phys_g in phys_gpus:
    tf.config.experimental.set_memory_growth(phys_g, True)

from default_config.util import *
from LSResNet import LSResNet
from get_data import get_data
#############################################
continue_training = (len(sys.argv) > 1) and (sys.argv[1] == 'continue')
#read_metrics = False
#############################################

#params = masif_opts["ligand"]
params = masif_opts["ligand_site"]

modelDir = 'kerasModel'
ckpPath = os.path.join(modelDir, 'ckp')

modelPath_endTraining = os.path.join(modelDir, 'savedModel_endTraining')

#############################################
#############################################
num_epochs = 40                 #############
starting_epoch = 15              ############
use_sample_weight = False        ############
#############################################
#############################################

model = LSResNet(
    params["max_distance"],
    feat_mask=params["feat_mask"],
    n_thetas=4,
    n_rhos=3,
    learning_rate = 1e-4,
    n_rotations=4,
    reg_val = 0
)

def hinge_accuracy(y_true, y_pred):
    y_true = tf.squeeze(y_true) > 0.0
    y_pred = tf.squeeze(y_pred) > 0.0
    result = tf.cast(y_true == y_pred, tf.float32)
    return tf.reduce_mean(result)

def F1(y_true, y_pred):
    y_true = tf.squeeze(y_true) > 0.0
    y_pred = tf.squeeze(y_pred) > 0.0
    overlap = tf.reduce_sum(tf.cast(y_true & y_pred, dtype=tf.float32))
    n_true = tf.reduce_sum(tf.cast(y_true, dtype=tf.float32))
    n_pred = tf.reduce_sum(tf.cast(y_pred, dtype=tf.float32))
    recall = overlap/n_true
    precision = overlap/n_pred
    return 2*precision*recall / (precision + recall)

binAcc = tf.keras.metrics.BinaryAccuracy(threshold = 0)
model.compile(optimizer = 'adam',
  loss = 'hinge',
  metrics=[hinge_accuracy, F1]
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
    best_acc = 0'''

training_list = np.load('/home/daniel.monyak/software/masif/data/masif_ligand/lists/train_pdbs_sequence.npy')
val_list = np.load('/home/daniel.monyak/software/masif/data/masif_ligand/lists/val_pdbs_sequence.npy')

#######################################
#######################################
#######################################

i = starting_epoch

print(f'Running training data, epoch {i}')
for i in range(num_epochs):
    if i < starting_epoch:
        continue
    
    train_j = 0
    #############################################################
    ###################     TRAINING DATA     ###################
    #############################################################
    for pdb_id in training_list:
        data = get_data(pdb_id)
        if data is None:
            continue
        
        X, y = data
        
        print(f'Epoch {i}, train pdb {train_j}, {pdb_id}')
        
        # TRAIN MODEL
        ################################################
        model.fit(X, y, verbose = 2)
        ################################################

        train_j += 1
    
    loss_list = []
    acc_list = []
    F1_list = []
    for pdb_id in val_list:
        data = get_data(pdb_id)
        if data is None:
            continue
            
        X, y = data
        loss, acc, F1 = model.evaluate(X, y, verbose=0)[:3]
        loss_list.append(loss)
        acc_list.append(acc)
        F1_list.append(F1)
    
    print(f'Epoch {i}, Validation Metrics')
    print(f'Loss: {np.mean(loss_list)}')
    print(f'Accuracy: {np.mean(acc_list)}')
    print(f'F1: {np.mean(F1_list)}')
    
    print(f'Saving model weights to {ckpPath}')
    model.save_weights(ckpPath)

print(f'Finished {num_epochs} training epochs!')
