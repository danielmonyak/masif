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
from tf2.masif_ligand.stochastic.MaSIF_ligand import MaSIF_ligand
from tf2.masif_ligand.stochastic.get_data import get_data

#from time import time

params = masif_opts["ligand"]

lr = 1e-4

n_train_batches = 10
batch_sz = 64
n_val = 50

reg_val = 0.0
reg_type = 'l2'

dev = '/GPU:3'

minPockets = params['minPockets']

train_list = np.load('/home/daniel.monyak/software/masif/data/masif_ligand/newPDBs/lists/train_reg.npy')
val_list = np.load('/home/daniel.monyak/software/masif/data/masif_ligand/newPDBs/lists/val_reg.npy')

np.random.shuffle(train_list)
train_iter = iter(train_list)
val_iter = iter(val_list)

modelDir = 'kerasModel'
modelPath = os.path.join(modelDir, 'savedModel')

##########################################
##########################################
with open('train_vars.pickle', 'rb') as handle:
    train_vars = pickle.load(handle)

continue_training = train_vars['continue_training']
ckpPath = train_vars['ckpPath']
num_iterations = train_vars['num_iterations']
starting_iteration = train_vars['starting_iteration']

print(f'Training for {num_iterations} iterations')
if continue_training:
    print(f'Resuming training from checkpoint at {ckpPath}, starting at iteration {starting_iteration}')

##########################################
##########################################

include_solvents = False

if include_solvents:
    ligand_list = masif_opts['all_ligands']
else:
    ligand_list = masif_opts['ligand_list']

model = MaSIF_ligand(
    params["max_distance"],
    len(ligand_list),
    feat_mask=params["feat_mask"],
    reg_val = reg_val, reg_type = reg_type,
    keep_prob=1.0
)
if continue_training:
    model.load_weights(ckpPath)
    print(f'Loaded model from {ckpPath}')
print()

optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_metric = tf.keras.metrics.Mean()

train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

grads = None
y_true_idx_used = np.zeros(len(ligand_list))

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)
    loss_metric.update_state(loss_value)
    
    train_acc_metric.update_state(y, logits)
    return tape.gradient(loss_value, model.trainable_weights)
    #return loss_value, tape.gradient(loss_value, model.trainable_weights)
    #return loss_value


@tf.function
def apply_gradient(grads):
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    
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

#        get_data_time = 0
#        train_time = 0
#        batch_time = -time()

        while j < n_train_batches:

#            get_data_time -= time()

            try:
                pdb_id = next(train_iter)
            except:
                np.random.shuffle(train_list)
                train_iter = iter(train_list)
                print('\nReshuffling training set...')
                continue

            data = get_data(pdb_id, include_solvents)


#            get_data_time += time()

            if data is None:
                continue

#            train_time -= time()

            X, pocket_points, y = data
            for k, pp in enumerate(pocket_points):
                pp_rand = np.random.choice(pp, minPockets, replace=False)
                X_temp = tuple(tf.constant(arr[:, pp_rand]) for arr in X)
                y_temp = tf.constant(y[k])

                grads = train_step(X_temp, y_temp)

                #loss_value, grads = train_step(X_temp, y_temp)
                #loss_list.append(loss_value)

                y_true_idx_used[y[k]] = 1

                if i == 0:
                    grads_sum = grads
                else:
                    grads_sum = [grads_sum[grad_i]+grads[grad_i] for grad_i in range(len(grads))]

                i += 1
                iterations += 1
            pdb_count += 1

#            train_time += time()

            if (i >= batch_sz) and (np.mean(y_true_idx_used) > 0.8):
                print(f'Training batch {j} - {i} pockets')

                mean_loss = float(loss_metric.result())
                train_acc = float(train_acc_metric.result())
                loss_metric.reset_states()
                train_acc_metric.reset_states()
                print("Loss -------- %.4f, Accuracy -------- %.4f, %d total PDBs" % (mean_loss, train_acc, pdb_count))

#                before = time()
                grads = [tsr/i for tsr in grads_sum]
#                after = time()
#                print(f'Mean grads: %.4f' % (after-before))

#                before = time()
                prep = zip(grads, model.trainable_weights)
#                after = time()
#                print(f'prep: %.4f' % (after-before))

#                before = time()
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
                #apply_gradient(grads)
#                after = time()
#                print('Apply gradients: %.3f' % (after-before))

#                print('get_data_time: %.4f' % get_data_time)
#                get_data_time = 0

#                print('train_time: %.4f' % train_time)
#                train_time = 0

                i = 0
                pdb_count = 0
                y_true_idx_used.fill(0)
                j += 1

#                batch_time += time()
#                print('batch_time: %.4f' % batch_time)
#                batch_time = -time()

        '''
        #mean_loss = np.mean(loss_list)
        #loss_list = []
        mean_loss = float(loss_metric.result())
        train_acc = train_acc_metric.result()

        loss_metric.reset_states()
        train_acc_metric.reset_states()

        print(f'\nTRAINING results over {j} batches, {pdb_count} total PDBs') 
        print("Loss --------------------- %.4f" % (mean_loss,))
        print("Accuracy ----------------- %.4f" % (float(train_acc),))'''

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

            data = get_data(pdb_id, include_solvents)
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

        val_acc = val_acc_metric.result()

        print(f'\nVALIDATION results over {i} pockets from {pdb_count} PDBs') 
        print('Accuracy ----------------- %.4f' % (float(val_acc),))

        val_acc_metric.reset_states()

        print(f'Saving model weights to {ckpPath}\n')
        model.save_weights(ckpPath)

model.save(modelPath)
