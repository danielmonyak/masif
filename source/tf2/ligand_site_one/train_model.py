import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import sys
import numpy as np
from scipy import spatial
import tensorflow as tf
import myMetrics

phys_gpus = tf.config.list_physical_devices('GPU')
for phys_g in phys_gpus:
    tf.config.experimental.set_memory_growth(phys_g, True)

import default_config.util as util
from default_config.masif_opts import masif_opts
from MaSIF_ligand_site_one import MaSIF_ligand_site
from get_data import get_data

params = masif_opts["ligand_site"]

#############################################
#############################################
lr = 1e-4

use_sample_weight = False        #############
#############################################
#############################################

from train_vars import train_vars

continue_training = train_vars['continue_training']
num_epochs = train_vars['num_epochs']
starting_epoch = train_vars['starting_epoch']
ckpPath = train_vars['ckpPath']

print(f'Training for {num_epochs} epochs')
if continue_training:
    print(f'Resuming training from checkpoint at {ckpPath}, starting at epoch {starting_epoch}')

model = MaSIF_ligand_site(
    params["max_distance"],
    feat_mask=params["feat_mask"],
    n_thetas=4,
    n_rhos=3,
    learning_rate = lr,
    n_rotations=4,
    reg_val = 0
)

def F1_lower(y_true, y_pred): return util.F1(y_true, y_pred, threshold=0.4)
def F1_upper(y_true, y_pred): return util.F1(y_true, y_pred, threshold=0.6)

from_logits = model.loss_fn.get_config()['from_logits']
thresh = (not from_logits) * 0.5
binAcc = tf.keras.metrics.BinaryAccuracy(threshold = thresh)
auc = tf.keras.metrics.AUC(from_logits = from_logits)

model.compile(optimizer = model.opt,
  loss = model.loss_fn,
  metrics=[binAcc, auc, F1_lower, F1, F1_upper]
)

if continue_training:
    model.load_weights(ckpPath)
    print(f'Loaded model from {ckpPath}')

training_list = np.load('/home/daniel.monyak/software/masif/data/masif_ligand/lists/train_pdbs_sequence.npy')
val_list = np.load('/home/daniel.monyak/software/masif/data/masif_ligand/lists/val_pdbs_sequence.npy')

#######################################
#######################################
#######################################

i = starting_epoch

print(f'Running training data, epoch {i}')
for i in range(num_epochs):
    train_j = 0
    
    np.random.shuffle(training_list)
    #############################################################
    ###################     TRAINING DATA     ###################
    #############################################################
    for pdb_id in training_list:
        data = get_data(pdb_id)
        if data is None:
            continue
        
        X, y, sample_weight = data
        if not use_sample_weight:
            sample_weight = None
        
        print(f'Epoch {i}, train pdb {train_j}, {pdb_id}')
        
        # TRAIN MODEL
        ################################################
        model.fit(X, y, verbose = 2,
                  sample_weight = sample_weight)
        ################################################

        train_j += 1
        
    loss_list = []
    acc_list = []
    auc_list = []
    F1_lower_list = []
    F1_list = []
    F1_upper_list = []
    for pdb_id in val_list:
        data = get_data(pdb_id)
        if data is None:
            continue
        
        if len(data) == 3:
            X, y, sample_weight = data
            sample_weight = sample_weight
        else:
            sample_weight = None
            X, y = data
        
        loss, acc, auc, F1_lower_, F1_, F1_upper_  = model.evaluate(X, y, verbose=0)[:6]
        loss_list.append(loss)
        acc_list.append(acc)
        auc_list.append(auc)
        F1_lower_list.append(F1_lower_)
        F1_list.append(F1_)
        F1_upper_list.append(F1_upper_)
    
    print(f'Epoch {i}, Validation Metrics')
    print(f'Loss: {np.mean(loss_list)}')
    print(f'Binary Accuracy: {np.mean(acc_list)}')
    print(f'AUC: {np.mean(auc_list)}')
    print(f'F1_lower_list: {np.mean(F1_lower_list)}')
    print(f'F1: {np.mean(F1_list)}')
    print(f'F1_upper_list: {np.mean(F1_upper_list)}')
    
    print(f'Saving model weights to {ckpPath}')
    model.save_weights(ckpPath)

print(f'Finished {num_epochs} training epochs!')
