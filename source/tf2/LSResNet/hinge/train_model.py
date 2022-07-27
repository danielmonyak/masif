import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import sys
import numpy as np
import tensorflow as tf

phys_gpus = tf.config.list_physical_devices('GPU')
for phys_g in phys_gpus:
    tf.config.experimental.set_memory_growth(phys_g, True)

import default_config.util as util
from default_config.masif_opts import masif_opts
from LSResNet import LSResNet
#from LSResNet_deep import LSResNet
from get_data import get_data

#params = masif_opts["ligand"]
params = masif_opts["LSResNet"]

modelDir = 'kerasModel'
ckpPath = os.path.join(modelDir, 'ckp')

modelPath_endTraining = os.path.join(modelDir, 'savedModel_endTraining')

#############################################
#############################################
lr = 1e-2

use_sample_weight = False        ############
train_batch_sz_threshold = 10   #############
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

model = LSResNet(
    params["max_distance"],
    feat_mask=params["feat_mask"],
    n_thetas=4,
    n_rhos=3,
    learning_rate = lr,
    n_rotations=4,
    reg_val = 0,
    use_special_neuron = True
)
model.compile(optimizer = 'adam')

'''
hinge_inst = losses.Hinge()
hinge_p = 3
def polynomialHingeLoss(y, y_pred):
    return tf.pow(hinge_inst(y, y_pred), hinge_p)

model.compile(optimizer = 'adam',
             loss = polynomialHingeLoss,
             metrics = [F1, )
'''

if continue_training:
    model.load_weights(ckpPath)
    print(f'Loaded model from {ckpPath}')

training_list = np.load('/home/daniel.monyak/software/masif/data/masif_ligand/lists/train_pdbs_sequence.npy')
val_list = np.load('/home/daniel.monyak/software/masif/data/masif_ligand/lists/val_pdbs_sequence.npy')

#######################################
#######################################
#######################################

i = starting_epoch

cur_batch_sz = 0
print(f'Running training data, epoch {i}')
for i in range(num_epochs):
    if i < starting_epoch:
        continue
    
    train_j = 0
    batch_i = 0
    #############################################################
    ###################     TRAINING DATA     ###################
    #############################################################
    
    np.random.shuffle(training_list)
    
    loss_list = []
    acc_list = []
    F1_list = []
    for pdb_id in training_list:
        data = get_data(pdb_id)
        
        if data is None:
            continue
        
        #X, y = data
        dataset_temp = tf.data.Dataset.from_tensors(data)
        if cur_batch_sz == 0:
            dataset = dataset_temp
            cur_batch_sz += 1
        else:
            dataset = dataset.concatenate(dataset_temp)
            cur_batch_sz += 1
            if cur_batch_sz == train_batch_sz_threshold:
                print(f'Epoch {i}, training on {cur_batch_sz} pdbs, batch {batch_i}')
                
                history = model.fit(dataset, verbose = 2)
                loss_list.extend(history.history['loss'])
                acc_list.extend(history.history['hinge_accuracy'])
                F1_list.extend(history.history['F1'])
                
                cur_batch_sz = 0
                batch_i += 1
        
        train_j += 1
    
    if cur_batch_sz > 0:
        print(f'Epoch {i}, training on {cur_batch_sz} pdbs, batch {batch_i}')
        
        history = model.fit(dataset, verbose = 2)
        loss_list.extend(history.history['loss'])
        acc_list.extend(history.history['hinge_accuracy'])
        F1_list.extend(history.history['F1'])
        
        cur_batch_sz = 0
    
    print(f'\nEpoch {i}, Training Metrics')
    print(f'Loss: {np.mean(loss_list)}')
    print(f'Hinge Accuracy: {np.mean(acc_list)}')
    print(f'F1: {np.mean(F1_list)}\n')
    
    loss_list = []
    acc_list = []
    F1_list = []
    for pdb_id in val_list:
        data = get_data(pdb_id)
        if data is None:
            continue
            
        X, y = data
        loss, f1, acc = model.evaluate(X, y, verbose=0)
        loss_list.append(loss)
        acc_list.append(acc)
        F1_list.append(f1)
    
    print(f'\nEpoch {i}, Validation Metrics')
    print(f'Loss: {np.mean(loss_list)}')
    print(f'Hinge Accuracy: {np.mean(acc_list)}')
    print(f'F1: {np.mean(F1_list)}\n')
    
    print(f'Saving model weights to {ckpPath}')
    model.save_weights(ckpPath)

print(f'Finished {num_epochs} training epochs!')
