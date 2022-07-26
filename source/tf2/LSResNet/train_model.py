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
#from LSResNet_deep import LSResNet
from LSResNet import LSResNet
from get_data import get_data
#############################################
#continue_training = (len(sys.argv) > 1) and (sys.argv[1] == 'continue')
#############################################

params = masif_opts["LSResNet"]



#############################################
#############################################
#num_epochs = 200                #############
#starting_epoch = 0              #############
use_sample_weight = False       #############
train_batch_sz_threshold = 32   #############
#############################################
#############################################

continue_key = input('Continue training from checkpoint? (y/[n]): ')
if (continue_key == '') or (continue_key == 'n'):
    continue_training = False
elif continue_key == 'y':
    continue_training = True
else:
    sys.exit('Please enter a valid choice...')

if continue_training:
    ckpPath = os.path.join('kerasModel', 'ckp')
    ckpKey = input(f'Using checkpoint at {ckpPath}? ([y]/n): ')
    if (ckpKey != '') and (ckpKey != 'y'):
        if ckpKey == 'n':
            ckpPath = input('Enter checkpoint path: ')
        else:
            sys.exit('Please enter a valid choice...')
    
    starting_epoch = int(input('Starting epoch: '))
else:
    ckpPath = os.path.join('kerasModel', 'ckp')
    starting_epoch = 0

num_epochs = int(input('Enter the number of epochs to train for: '))


print(f'Training for {num_epochs} epochs')
if continue_training:
    print(f'Resuming training from checkpoint at {ckpPath}, starting at epoch {starting_epoch}')


model = LSResNet(
    params["max_distance"],
    feat_mask=params["feat_mask"],
    n_thetas=4,
    n_rhos=3,
    learning_rate = 1e-4,
    n_rotations=4,
    reg_val = 0
)

def F1_04(y_true, y_pred): return F1(y_true, y_pred, threshold=0.4)
def F1_06(y_true, y_pred): return F1(y_true, y_pred, threshold=0.6)

from_logits = model.loss_fn.get_config()['from_logits']
binAcc = tf.keras.metrics.BinaryAccuracy(threshold = (not from_logits) * 0.5)
auc = tf.keras.metrics.AUC(from_logits = from_logits)
model.compile(optimizer = model.opt,
  loss = model.loss_fn,
  metrics=[binAcc, auc, F1_04, F1, F1_06]
)

code32 = tf.cast(params['defaultCode'], dtype=tf.float32)
code64 = tf.cast(params['defaultCode'], dtype=tf.float64)
dataset_padding = (((code64, code64, code64, code64), code64), code32)

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
    train_j = 0
    batch_i = 0
    #############################################################
    ###################     TRAINING DATA     ###################
    #############################################################
    
    np.random.shuffle(training_list)
    
    loss_list = []
    acc_list = []
    auc_list = []
    F1_04_list = []
    F1_list = []
    F1_06_list = []
    for pdb_id in training_list:
        data = get_data(pdb_id)
        if data is None:
            continue
        
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
                acc_list.extend(history.history['binary_accuracy'])
                auc_list.extend(history.history['auc'])
                F1_04_list.extend(history.history['F1_04'])
                F1_list.extend(history.history['F1'])
                F1_06_list.extend(history.history['F1_06'])
                
                cur_batch_sz = 0
                batch_i += 1
        
        train_j += 1
    
    if cur_batch_sz > 0:
        print(f'Epoch {i}, training on {cur_batch_sz} pdbs, batch {batch_i}')
        
        history = model.fit(dataset, verbose = 2)
        loss_list.extend(history.history['loss'])
        acc_list.extend(history.history['binary_accuracy'])
        auc_list.extend(history.history['auc'])
        F1_04_list.extend(history.history['F1_04'])
        F1_list.extend(history.history['F1'])
        F1_06_list.extend(history.history['F1_06'])
        
        cur_batch_sz = 0
    
    print(f'\nEpoch {i}, Training Metrics')
    print(f'Loss: {np.mean(loss_list)}')
    print(f'Binary Accuracy: {np.mean(acc_list)}')
    print(f'AUC: {np.mean(auc_list)}')
    print(f'F1_04: {np.mean(F1_04_list)}\n')
    print(f'F1: {np.mean(F1_list)}\n')
    print(f'F1_06: {np.mean(F1_06_list)}\n')
    
    loss_list = []
    acc_list = []
    auc_list = []
    F1_04_list = []
    F1_list = []
    F1_06_list = []
    for pdb_id in val_list:
        data = get_data(pdb_id)
        if data is None:
            continue
            
        X, y = data
        loss, acc, auc, F1_04_, F1_, F1_06_ = model.evaluate(X, y, verbose=0)[:6]
        loss_list.append(loss)
        acc_list.append(acc)
        auc_list.append(auc)
        F1_04_list.append(F1_04_)
        F1_list.append(F1_)
        F1_06_list.append(F1_06_)
    
    print(f'\nEpoch {i}, Validation Metrics')
    print(f'Loss: {np.mean(loss_list)}')
    print(f'Binary Accuracy: {np.mean(acc_list)}')
    print(f'AUC: {np.mean(auc_list)}')
    print(f'F1_04: {np.mean(F1_04_list)}\n')
    print(f'F1: {np.mean(F1_list)}\n')
    print(f'F1_06: {np.mean(F1_06_list)}\n')
    
    print(f'Saving model weights to {ckpPath}')
    model.save_weights(ckpPath)

print(f'Finished {num_epochs} training epochs!')
