import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import sys
import numpy as np
import tensorflow as tf

phys_gpus = tf.config.list_physical_devices('GPU')
for phys_g in phys_gpus:
    tf.config.experimental.set_memory_growth(phys_g, True)

from default_config.util import *
#from LSResNet import LSResNet
from LSResNet_deep import LSResNet
from get_data import get_data
#############################################
continue_training = (len(sys.argv) > 1) and (sys.argv[1] == 'continue')
#read_metrics = False
#############################################

#params = masif_opts["ligand"]
params = masif_opts["LSResNet"]

modelDir = 'kerasModel'
ckpPath = os.path.join(modelDir, 'ckp')

modelPath_endTraining = os.path.join(modelDir, 'savedModel_endTraining')

#############################################
#############################################
num_epochs = 100                 #############
starting_epoch = 40               ############
use_sample_weight = False        ############
train_batch_sz_threshold = 10   #############
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

binAcc = tf.keras.metrics.BinaryAccuracy(threshold = 0)
model.compile(optimizer = 'adam',
  loss = 'squared_hinge',
  metrics=[hinge_accuracy, F1]
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
                model.fit(dataset, verbose = 2)
                cur_batch_sz = 0
                batch_i += 1
        
        #print(f'Epoch {i}, train pdb {train_j}, {pdb_id}')
        
        # TRAIN MODEL
        ################################################
        #model.fit(X, y, verbose = 2)
        ################################################

        train_j += 1
    
    if cur_batch_sz > 0:
        print(f'Epoch {i}, training on {cur_batch_sz} pdbs, batch {batch_i}')
        model.fit(dataset, verbose = 2)
        cur_batch_sz = 0
        
    print(f'Epoch {i}, calculating validation metrics...')
    
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
    
    
    print(f'Loss: {np.mean(loss_list)}')
    print(f'Accuracy: {np.mean(acc_list)}')
    print(f'F1: {np.nanmean(F1_list)}')
    
    print(f'Saving model weights to {ckpPath}')
    model.save_weights(ckpPath)

print(f'Finished {num_epochs} training epochs!')
