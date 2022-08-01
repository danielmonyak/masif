import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import sys
import numpy as np
from scipy import spatial
import tensorflow as tf

phys_gpus = tf.config.list_physical_devices('GPU')
for phys_g in phys_gpus:
    tf.config.experimental.set_memory_growth(phys_g, True)

import default_config.util as util
from default_config.masif_opts import masif_opts
from tf2.LSResNet.LSResNet import LSResNet
from get_data import get_data

params = masif_opts["LSResNet"]

#lr = 1e-3
lr = 1e-4

n_train = 300
n_val = 50


train_list = np.load('/home/daniel.monyak/software/masif/source/tf2/masif_ligand/stochastic/lists/train_pdbs.npy')
val_list = np.load('/home/daniel.monyak/software/masif/source/tf2/masif_ligand/stochastic/lists/val_pdbs.npy')

np.random.shuffle(train_list)
train_iter = iter(train_list)
val_iter = iter(val_list)

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

#with tf.device('/GPU:1'):
model = LSResNet(
    params["max_distance"],
    feat_mask=params["feat_mask"],
    n_thetas=4,
    n_rhos=3,
    learning_rate = lr,
    n_rotations=4,
    reg_val = 0,
    extra_conv_layers = False
)
'''model.compile(optimizer = model.opt,
  loss = model.loss_fn,
  metrics=[binAcc, auc, F1_04, util.F1, F1_06]
)'''
if continue_training:
    model.load_weights(ckpPath)
    print(f'Loaded model from {ckpPath}')

from_logits = True
optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=from_logits)


train_acc_metric = tf.keras.metrics.BinaryAccuracy(threshold = (not from_logits) * 0.5)
train_auc_metric = tf.keras.metrics.AUC(from_logits=from_logits)
train_F1_lower_metric = util.F1_Metric(from_logits=from_logits, threshold = 0.3)
train_F1_metric = util.F1_Metric(from_logits=from_logits, threshold = 0.5)

val_acc_metric = tf.keras.metrics.BinaryAccuracy(threshold = (not from_logits) * 0.5)
val_auc_metric = tf.keras.metrics.AUC(from_logits=from_logits)
val_F1_lower_metric = util.F1_Metric(from_logits=from_logits, threshold = 0.3)
val_F1_metric = util.F1_Metric(from_logits=from_logits, threshold = 0.5)

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_acc_metric.update_state(y, logits)
    train_auc_metric.update_state(y, logits)
    train_F1_lower_metric.update_state(y, logits)
    train_F1_metric.update_state(y, logits)
    return loss_value

@tf.function
def test_step(x, y):
    val_logits = model(x, training=False)
    val_acc_metric.update_state(y, val_logits)
    val_acc_metric.update_state(y, logits)
    val_auc_metric.update_state(y, logits)
    val_F1_lower_metric.update_state(y, logits)
    val_F1_metric.update_state(y, logits)

iterations = starting_iteration
while iterations < num_iterations:
    i = 0
    loss_list = []
    while i < n_train:
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
            
        X, y = data
        loss_value = train_step(X, y)
        loss_list.append(loss_value)
        i += 1
        iterations += 1
    
    mean_loss = np.mean(loss_list)
    train_acc = train_acc_metric.result()
    train_auc = train_auc_metric.result()
    train_F1_lower = train_F1_lower_metric.result()
    train_F1 = train_F1_metric.result()
    
    print(f'\nTRAINING results over {i} PDBs') 
    print("Loss --------------------- %.4f" % (mean_loss,))
    print("Accuracy ----------------- %.4f" % (float(train_acc),))
    print("AUC      ----------------- %.4f" % (float(train_auc),))
    print("F1 Lower ----------------- %.4f" % (float(train_F1_lower),))
    print("F1       ----------------- %.4f" % (float(train_F1),))
    
    print(f'{iterations} iterations completed')
    
    train_acc_metric.reset_states()
            
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
