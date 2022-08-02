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
from tf2.masif_ligand.stochastic.MaSIF_ligand import MaSIF_ligand
from tf2.masif_ligand.stochastic.get_data import get_data

params = masif_opts["ligand"]

lr = 1e-2

n_train_batches = 10
batch_sz = 32
n_val = 50

reg_val = 1e-4
reg_type = 'l2'

dev = '/GPU:3'

minPockets = params['minPockets']

train_list = np.load('../stochastic/lists/train_pdbs.npy')
val_list = np.load('../stochastic/lists/val_pdbs.npy')

np.random.shuffle(train_list)
train_iter = iter(train_list)
val_iter = iter(val_list)

modelDir = 'kerasModel'
modelPath = os.path.join(modelDir, 'savedModel')

##########################################
##########################################
from train_vars import train_vars

continue_training = train_vars['continue_training']
ckpPath = train_vars['ckpPath']
num_iterations = train_vars['num_iterations']
starting_iteration = train_vars['starting_iteration']

print(f'Training for {num_iterations} iterations')
if continue_training:
    print(f'Resuming training from checkpoint at {ckpPath}, starting at iteration {starting_iteration}')

##########################################
##########################################

model = MaSIF_ligand(
    params["max_distance"],
    len(masif_opts['all_ligands']),
    feat_mask=params["feat_mask"],
    reg_val = reg_val, reg_type = reg_type,
    keep_prob=1.0
)
if continue_training:
    model.load_weights(ckpPath)
    print(f'Loaded model from {ckpPath}')

optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

grads = None

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)
    return loss_value, tape.gradient(loss_value, model.trainable_weights)
    #return loss_value

@tf.function
def test_step(x, y):
    val_logits = model(x, training=False)
    val_acc_metric.update_state(y, val_logits)

iterations = starting_iteration
while iterations < num_iterations:
    i = 0
    j = 0
    pdb_count = 0
    loss_list = []
    while j < n_train_batches:
        try:
            pdb_id = next(train_iter)
        except:
            np.random.shuffle(train_list)
            train_iter = iter(train_list)
            print('\nReshuffling training set...')
            continue
        
        data = get_data(pdb_id)
        if data is None:
            continue
            
        X, pocket_points, y = data
        for k, pp in enumerate(pocket_points):
            pp_rand = np.random.choice(pp, minPockets, replace=False)
            X_temp = tuple(tf.constant(arr[:, pp_rand]) for arr in X)
            y_temp = tf.constant(y[k])
            grads, loss_value = train_step(X_temp, y_temp)
            loss_list.append(loss_value)
            
            if i == 0:
                grads_sum = grads
            else:
                grads_sum += grads
            
            i += 1
            iterations += 1
        pdb_count += 1
        
        if i >= batch_sz:
            print(f'Training batch {j} - {i} pockets')
            grads = grads_sum/i
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            train_acc_metric.update_state(y, logits)
            i = 0
            j += 1
    
    
    mean_loss = np.mean(loss_list)
    train_acc = train_acc_metric.result()
    
    print(f'\nTRAINING results over {j} batches, {pdb_count} total PDBs') 
    print("Loss --------------------- %.4f" % (mean_loss,))
    print("Accuracy ----------------- %.4f" % (float(train_acc),))
    
    print(f'{iterations} iterations completed')
    
    train_acc_metric.reset_states()
    
    #####################################
    #####################################
    i = 0
    pdb_count = 0
    while i < n_val:
        try:
            pdb_id = next(val_iter)
        except:
            val_iter = iter(val_list)
            continue
        
        data = get_data(pdb_id)
        if data is None:
            continue
            
        X, pocket_points, y = data
        for k, pp in enumerate(pocket_points):
            pp_rand = np.random.choice(pp, minPockets, replace=False)
            X_temp = tuple(tf.constant(arr[:, pp_rand]) for arr in X)
            y_temp = tf.constant(y[k])
            test_step(X_temp, y_temp)
            
            i += 1
        pdb_count += 1
    
    train_acc = val_acc_metric.result()
    
    print(f'\nVALIDATION results over {i} pockets from {pdb_count} PDBs') 
    print('Accuracy ----------------- %.4f' % (float(train_acc),))
    
    loss_list = []
    val_acc_metric.reset_states()
    
    print(f'Saving model weights to {ckpPath}')
    model.save_weights(ckpPath)
        
        
model.save(modelPath)
