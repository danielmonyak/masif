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
from tf2.LSResNet.LSResNet import LSResNet
from tf2.LSResNet.get_data import get_data

params = masif_opts["LSResNet"]

#############################################
#############################################

# Reularization coefficient
reg_val = 0.0
#lr = 1e-5

dev = '/GPU:0'
#############################################
#############################################
train_list = np.load('/home/daniel.monyak/software/masif/data/masif_ligand/newPDBs/lists/train_reg.npy')
val_list = np.load('/home/daniel.monyak/software/masif/data/masif_ligand/newPDBs/lists/val_reg.npy')

shared_old = np.loadtxt('/home/daniel.monyak/software/masif/data/masif_ligand/newPDBs/shared_old.txt', dtype=str)

##########################################
##########################################
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
model = LSResNet(
    params["max_distance"],
    feat_mask=params["feat_mask"],
    n_thetas=4,
    n_rhos=3,
    n_rotations=4,
    reg_val = reg_val,
    extra_conv_layers = False
)
if continue_training:
    model.load_weights(ckpPath)
    print(f'Loaded model from {ckpPath}')
else:
    with open('loss.txt', 'w') as f:
        pass

print()

#optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

acc_metric = tf.keras.metrics.BinaryAccuracy(threshold = 0, name = 'acc')
auc_metric = tf.keras.metrics.AUC(from_logits = True, name = 'auc')
F1_lowest_metric = util.F1_Metric(from_logits = True, threshold = 0.3, name = 'F1_lowest')
F1_lower_metric = util.F1_Metric(from_logits = True, threshold = 0.4, name = 'F1_lower')
F1_metric = util.F1_Metric(from_logits = True, threshold = 0.5, name = 'F1')

model.compile(
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9),
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics = [acc_metric, auc_metric, F1_lowest_metric, F1_lower_metric, F1_metric]
)

with tf.device(dev):
    iterations = starting_iteration
    epochs = 0
    while iterations < num_iterations:
        loss_list = []
        acc_list = []
        auc_list = []
        F1_lowest_list = []
        F1_lower_list = []
        F1_list = []
        
        np.random.shuffle(train_list)
        train_iter = iter(train_list)
        i = 0
        while True:
            try:
                pdb_id = next(train_iter)
            except:
                break

            ##### Training the model just on the old data - delete this
            if pdb_id not in shared_old:
                continue

            # Get y values from file
            try:
                y = np.load(os.path.join(params['masif_precomputation_dir'], pdb_id, 'LSRN_y.npy'))
            except:
                continue

            # Get X values from get_data
            data = get_data(pdb_id, training=True, make_y = False)
            if data is None:
                continue

            # Only care about X
            X, _ = data
            
            # Skip PDB if there are NaN values in input_feat - messes up training
            if np.sum(np.isnan(X[0][0])) > 0:
                continue
                
            X_tf = (tuple(tf.constant(arr) for arr in X[0]), tf.constant(X[1]))
            y_tf = tf.constant(y)

            history = model.fit(X_tf, y_tf, verbose=0, epochs=1)
            i += 1
            iterations += 1
            
            loss_list.extend(history.history['loss'])
            acc_list.extend(history.history['acc'])
            auc_list.extend(history.history['auc'])
            F1_lowest_list.extend(history.history['F1_lowest'])
            F1_lower_list.extend(history.history['F1_lower'])
            F1_list.extend(history.history['F1'])
            
            with open('loss.txt', 'a') as f:
                f.write(str(history.history['loss'][0]) + '\n')
        
        print(f'Results from {i} training samples')
        print("Loss -------------------- %.4f" % np.mean(loss_list))
        print("Accuracy ----------------- %.4f" % np.mean(acc_list))
        print("AUC      ----------------- %.4f" % np.mean(auc_list))
        print("F1 Lowest ----------------- %.4f" % np.mean(F1_lowest_list))
        print("F1 Lower ----------------- %.4f" % np.mean(F1_lower_list))
        print("F1       ----------------- %.4f" % np.mean(F1_list))
        
        epochs += 1
        print(f'\n{iterations} iterations, {epochs} epochs completed')
        
        ##################################### VALIDATION DATA
        #####################################
        acc_list = []
        auc_list = []
        F1_lowest_list = []
        F1_lower_list = []
        F1_list = []
        
        val_iter = iter(val_list)
        i = 0
        while True:
            try:
                pdb_id = next(val_iter)
            except:
                break
            
            ##### Evaluating the model just on the old data - delete this
            if pdb_id not in shared_old:
                continue

            try:
                y = np.load(os.path.join(params['masif_precomputation_dir'], pdb_id, 'LSRN_y.npy'))
            except:
                continue

            data = get_data(pdb_id, training=False, make_y = False)
            if data is None:
                continue
            X, _, _ = data

            X_tf = (tuple(tf.constant(arr) for arr in X[0]), tf.constant(X[1]))
            y_tf = tf.constant(y)

            _, acc, auc, F1_lowest, F1_lower, F1 = model.evaluate(X_tf, y_tf, verbose=0)
            acc_list.append(acc)
            auc_list.append(auc)
            F1_lowest_list.append(F1_lowest)
            F1_lower_list.append(F1_lower)
            F1_list.append(F1)
            
            i += 1

        print(f'\nVALIDATION results over {i} PDBs') 
        print("Accuracy ----------------- %.4f" % np.mean(acc_list))
        print("AUC      ----------------- %.4f" % np.mean(auc_list))
        print("F1 Lowest ----------------- %.4f" % np.mean(F1_lowest_list))
        print("F1 Lower ----------------- %.4f" % np.mean(F1_lower_list))
        print("F1       ----------------- %.4f" % np.mean(F1_list))

        print(f'\nSaving model weights to {ckpPath}\n')
        model.save_weights(ckpPath)
