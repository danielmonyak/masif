import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import sys
import numpy as np
from scipy import spatial
import tensorflow as tf

phys_gpus = tf.config.list_physical_devices('GPU')
for phys_g in phys_gpus:
    tf.config.experimental.set_memory_growth(phys_g, True)

from default_config.util import *
from LSResNet_deep import LSResNet
from get_data import get_data
#############################################
continue_training = (len(sys.argv) > 1) and (sys.argv[1] == 'continue')
#############################################

params = masif_opts["LSResNet"]

modelDir = 'kerasModel'
ckpPath = os.path.join(modelDir, 'ckp')

modelPath_endTraining = os.path.join(modelDir, 'savedModel_endTraining')

#############################################
#############################################
num_epochs = 40                 #############
starting_epoch = 0              #############
use_sample_weight = False        #############
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

from_logits = model.loss_fn.get_config()['from_logits']
thresh = (not from_logits) * 0.5
binAcc = tf.keras.metrics.BinaryAccuracy(threshold = thresh)
auc = tf.keras.metrics.AUC(from_logits = from_logits)
model.compile(optimizer = model.opt,
  loss = model.loss_fn,
  metrics=[binAcc, auc, F1]
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
    #############################################################
    ###################     TRAINING DATA     ###################
    #############################################################
    
    np.random.shuffle(training_list)
    
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
    auc_list = []
    F1_list = []
    for pdb_id in val_list:
        data = get_data(pdb_id)
        if data is None:
            continue
            
        X, y = data
        loss, acc, auc, F1 = model.evaluate(X, y, verbose=0)[:4]
        loss_list.append(loss)
        acc_list.append(acc)
        auc_list.append(auc)
        F1_list.append(F1)
    
    print(f'Epoch {i}, Validation Metrics')
    print(f'Loss: {np.mean(loss_list)}')
    print(f'Binary Accuracy: {np.mean(acc_list)}')
    print(f'AUC: {np.mean(auc_list)}')
    
    print(f'Saving model weights to {ckpPath}')
    model.save_weights(ckpPath)

print(f'Finished {num_epochs} training epochs!')
