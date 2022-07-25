import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import sys
import importlib
from IPython.core.debugger import set_trace
import pickle
import numpy as np
from scipy import spatial
import tensorflow as tf
import myMetrics

phys_gpus = tf.config.list_physical_devices('GPU')
for phys_g in phys_gpus:
    tf.config.experimental.set_memory_growth(phys_g, True)

from default_config.util import *
from tf2.ligand_site_one.sep_layers.MaSIF_ligand_site_one import MaSIF_ligand_site
from get_data import get_data
#############################################
continue_training = False
#read_metrics = False
#############################################

#params = masif_opts["ligand"]
params = masif_opts["ligand_site"]

modelDir = 'kerasModel'
ckpPath = os.path.join(modelDir, 'ckp')
ckpStatePath = ckpPath + '.pickle'

modelPath_endTraining = os.path.join(modelDir, 'savedModel_endTraining')

#############################################
#############################################
num_epochs = 20                 #############
starting_epoch = 0              #############
use_sample_weight = True        #############
#############################################
#############################################

model = MaSIF_ligand_site(
    params["max_distance"],
    feat_mask=params["feat_mask"],
    n_thetas=4,
    n_rhos=3,
    learning_rate = 1e-4,
    n_rotations=4,
    reg_val = 0
)

from_logits = model.loss_fn.get_config()['from_logits']
thresh = (not from_logits) * 0.5
binAcc = tf.keras.metrics.BinaryAccuracy(threshold = thresh)
auc = tf.keras.metrics.AUC(from_logits = from_logits)
TP = myMetrics.TruePositives(from_logits = from_logits)
TN = myMetrics.TrueNegatives(from_logits = from_logits)
FP = myMetrics.FalsePositives(from_logits = from_logits)
FN = myMetrics.FalseNegatives(from_logits = from_logits)

model.compile(optimizer = model.opt,
  loss = model.loss_fn,
  metrics=[binAcc, auc, TP, FN]
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
    train_j = 0
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
        
        loss, acc, auc = model.evaluate(X, y, verbose=0)[:3]
        loss_list.append(loss)
        acc_list.append(acc)
        auc_list.append(auc)
    
    print(f'Epoch {i}, Validation Metrics')
    print(f'Loss: {np.mean(loss_list)}')
    print(f'Binary Accuracy: {np.mean(acc_list)}')
    print(f'AUC: {np.mean(auc_list)}')
    
    print(f'Saving model weights to {ckpPath}')
    model.save_weights(ckpPath)

print(f'Finished {num_epochs} training epochs!')
