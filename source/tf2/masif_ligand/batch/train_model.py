import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import sys
import numpy as np
import pickle
import tensorflow as tf

phys_gpus = tf.config.list_physical_devices('GPU')
for phys_g in phys_gpus:
    tf.config.experimental.set_memory_growth(phys_g, True)

import default_config.util as util
from default_config.masif_opts import masif_opts
from tf2.masif_ligand.batch.MaSIF_ligand import MaSIF_ligand
from tf2.masif_ligand.batch.get_data import get_data

params = masif_opts["ligand"]

#############################################
#############################################

# Reularization coefficient
reg_val = 0.0

n_train_batches = 10
#batch_sz = 64
batch_sz = 32
n_val = 50

dev = '/GPU:3'

# Whether or not to include the solvent PDBs
include_solvents = False
#############################################
#############################################
minPockets = params['minPockets']

train_list = np.load('/home/daniel.monyak/software/masif/data/masif_ligand/newPDBs/lists/train_reg.npy')
val_list = np.load('/home/daniel.monyak/software/masif/data/masif_ligand/newPDBs/lists/val_reg.npy')

np.random.shuffle(train_list)
train_iter = iter(train_list)
val_iter = iter(val_list)

modelDir = 'kerasModel'
modelPath = os.path.join(modelDir, 'savedModel')

##########################################
########################################## Gather input passed collected by input prompts
with open('train_vars.pickle', 'rb') as handle:
    train_vars = pickle.load(handle)

continue_training = train_vars['continue_training']
ckpPath = train_vars['ckpPath']
num_iterations = train_vars['num_iterations']
starting_iteration = train_vars['starting_iteration']
lr = train_vars['lr']

print(f'Training for {num_iterations} iterations, using learning rate {lr:.1e}')
if continue_training:
    print(f'Resuming training from checkpoint at {ckpPath}, starting at iteration {starting_iteration}, using learning rate {lr:.1e}')

##########################################
##########################################
if include_solvents:
    ligand_list = masif_opts['all_ligands']
else:
    ligand_list = masif_opts['ligand_list']

model = MaSIF_ligand(
    params["max_distance"],
    len(ligand_list),
    feat_mask=params["feat_mask"],
    reg_val = reg_val,
    keep_prob=1.0
)
if continue_training:
    model.load_weights(ckpPath)
    print(f'Loaded model from {ckpPath}')
else:
    with open('loss.txt', 'w') as f:
        pass
print()

optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
#optimizer = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True, clipvalue=0.5)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_metric = tf.keras.metrics.Mean()

train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

# Keeps track of if an example for each ligand class is currently in the training batch
y_true_idx_used = np.zeros(len(ligand_list))

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)
    loss_metric.update_state(loss_value)
    
    train_acc_metric.update_state(y, logits)
    return tape.gradient(loss_value, model.trainable_weights)

@tf.function
def test_step(x, y):
    val_logits = model(x, training=False)
    val_acc_metric.update_state(y, val_logits)

with tf.device(dev):

    iterations = starting_iteration
    while iterations < num_iterations:
        i = 0
        j = 0
        pdb_count = 0

        while j < n_train_batches:
            try:
                pdb_id = next(train_iter)
            except:
                np.random.shuffle(train_list)
                train_iter = iter(train_list)
                print('\nReshuffling training set...')
                continue
            
            try:
                X = np.load(os.path.join(params['masif_precomputation_dir'], pdb_id, 'X.npy'), allow_pickle=True)
                y = np.load(os.path.join(params['masif_precomputation_dir'], pdb_id, 'y.npy'))
            except:
                continue
            
            for k, X_temp in enumerate(X):
                # Don't use pocket if it is a solvent pocket (assuming include_solvents=True)
                if y[k] >= len(ligand_list):
                    continue
                
                # Skip PDB if there are NaN values in input_feat - messes up training
                if np.sum(np.isnan(X_temp[0])) > 0:
                    continue
                
                X_temp[0] = np.expand_dims(X_temp[0], axis=0)
                X_temp[3] = np.expand_dims(X_temp[3], axis=0)
                
                # Number of patches in this pocket
                n_samples = X_temp[0].shape[1]
                
                # Randomly sample 32 patches
                pp_rand = np.random.choice(range(n_samples), minPockets, replace=False)
                X_temp = tuple(tf.constant(arr[:, pp_rand]) for arr in X_temp)
                y_temp = tf.constant(y[k])
                
                grads = train_step(X_temp, y_temp)
                
                # Mark that a pocket with this ligand has been trained on
                y_true_idx_used[y[k]] = 1
                
                # Make sure gradients don't get NaN values
                skip = False
                for g in grads:
                    if np.any(np.isnan(g)):
                        skip = True
                        break
                if skip:
                    continue

                # If first pocket of batch, set grads, otherwise add to existing grads
                if i == 0:
                    grads_sum = grads
                else:
                    grads_sum = [grads_sum[grad_i]+grads[grad_i] for grad_i in range(len(grads))]
                i += 1
                iterations += 1
            pdb_count += 1
            
            # Once number of pockets in batch is past "batch_sz" and 80% of ligand classes are accounted for 
            if (i >= batch_sz) and (np.mean(y_true_idx_used) > 0.8):
            #if (i >= batch_sz):
                print(f'Training batch {j} - {i} pockets')
                mean_loss = float(loss_metric.result())
                train_acc = float(train_acc_metric.result())
                loss_metric.reset_states()
                train_acc_metric.reset_states()
                
                print("Loss -------- %.4f, Accuracy -------- %.4f, %d total PDBs" % (mean_loss, train_acc, pdb_count))
                
                # Save loss values to fle
                with open('loss.txt', 'a') as f:
                    f.write(str(mean_loss) + '\n')
                
                # Use average of all gradients
                grads = [tsr/i for tsr in grads_sum]
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
                
                i = 0
                pdb_count = 0
                y_true_idx_used.fill(0)
                j += 1

        print(f'{iterations} iterations completed')

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

            try:
                X = np.load(os.path.join(params['masif_precomputation_dir'], pdb_id, 'X.npy'), allow_pickle=True)
                y = np.load(os.path.join(params['masif_precomputation_dir'], pdb_id, 'y.npy'))
            except:
                continue
            
            for k, X_temp in enumerate(X):
                if y[k] >= len(ligand_list):
                    continue
                X_temp[0] = np.expand_dims(X_temp[0], axis=0)
                
                # Skip PDB if there are NaN values in input_feat - messes up training
                if np.sum(np.isnan(X_temp[0])) > 0:
                    continue
                
                X_temp[3] = np.expand_dims(X_temp[3], axis=0)
                n_samples = X_temp[0].shape[1]
                pp_rand = np.random.choice(range(n_samples), minPockets, replace=False)
                X_temp = tuple(tf.constant(arr[:, pp_rand]) for arr in X_temp)
                y_temp = tf.constant(y[k])
            
                test_step(X_temp, y_temp)

                i += 1
            pdb_count += 1

        val_acc = val_acc_metric.result()

        print(f'\nVALIDATION results over {i} pockets from {pdb_count} PDBs') 
        print('Accuracy ----------------- %.4f' % (float(val_acc),))

        val_acc_metric.reset_states()

        print(f'Saving model weights to {ckpPath}\n')
        model.save_weights(ckpPath)

model.save(modelPath)
