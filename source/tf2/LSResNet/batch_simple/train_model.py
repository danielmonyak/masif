import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import sys
import numpy as np
from scipy import spatial
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

lr = 1e-2

n_train_batches = 10
batch_sz = 32
n_val = 50

train_list = np.load('/home/daniel.monyak/software/masif/data/masif_ligand/newPDBs/lists/train_reg.npy')
val_list = np.load('/home/daniel.monyak/software/masif/data/masif_ligand/newPDBs/lists/val_reg.npy')

np.random.shuffle(train_list)
train_iter = iter(train_list)
val_iter = iter(val_list)

##########################################
##########################################
#from train_vars import train_vars
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
if continue_training:
    model.load_weights(ckpPath)
    print(f'Loaded model from {ckpPath}')
else:
    with open('loss.txt', 'w') as f:
        pass
print()

#optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
loss_metric = tf.keras.metrics.Mean()

train_acc_metric = tf.keras.metrics.BinaryAccuracy()
train_auc_metric = tf.keras.metrics.AUC()
train_F1_lower_metric = util.F1_Metric(threshold = 0.3)
train_F1_metric = util.F1_Metric(threshold = 0.5)

val_acc_metric = tf.keras.metrics.BinaryAccuracy()
val_auc_metric = tf.keras.metrics.AUC()
val_F1_lower_metric = util.F1_Metric(threshold = 0.3)
val_F1_metric = util.F1_Metric(threshold = 0.5)

grads = None

@tf.function(experimental_relax_shapes=True)
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)
    loss_metric.update_state(loss_value)
    
    y_pred = tf.sigmoid(logits)
    train_acc_metric.update_state(y, y_pred)
    train_auc_metric.update_state(y, y_pred)
    train_F1_lower_metric.update_state(y, y_pred)
    train_F1_metric.update_state(y, y_pred)
    
    return tape.gradient(loss_value, model.trainable_weights)

@tf.function(experimental_relax_shapes=True)
def test_step(x, y):
    logits = model(x, training=False)
    y_pred = tf.sigmoid(logits)
    val_acc_metric.update_state(y, y_pred)
    val_auc_metric.update_state(y, y_pred)
    val_F1_lower_metric.update_state(y, y_pred)
    val_F1_metric.update_state(y, y_pred)

    
with tf.device('/GPU:3'):
    iterations = starting_iteration
    epoch = 0
    while iterations < num_iterations:
        i = 0
        j = 0
        while j < n_train_batches:
            try:
                pdb_id = next(train_iter)
            except:
                np.random.shuffle(train_list)
                train_iter = iter(train_list)
                print('\nReshuffling training set...')
                epoch += 1
                continue

            try:
                y = np.load(os.path.join(params['masif_precomputation_dir'], pdb_id, 'LSRN_y.npy'))
            except:
                continue

            data = get_data(pdb_id, training=True, make_y = False)
            if data is None:
                continue
            X, _ = data

            X_tf = (tuple(tf.constant(arr) for arr in X[0]), tf.constant(X[1]))
            y_tf = tf.constant(y)

            grads = train_step(X_tf, y_tf)

            if i == 0:
                grads_sum = grads
            else:
                grads_sum = [grads_sum[grad_i]+grads[grad_i] for grad_i in range(len(grads))]

            i += 1
            iterations += 1

            if i >= batch_sz:
                mean_loss = float(loss_metric.result())
                train_acc = float(train_acc_metric.result())
                train_auc = float(train_auc_metric.result())
                train_F1_lower = float(train_F1_lower_metric.result())
                train_F1 = float(train_F1_metric.result())

                loss_metric.reset_states()
                train_acc_metric.reset_states()
                train_auc_metric.reset_states()
                train_F1_lower_metric.reset_states()
                train_F1_metric.reset_states()

                print(f'\nTraining batch {j} - {i} proteins')
                print("Loss --------------------- %.4f" % mean_loss)
                print("Accuracy ----------------- %.4f" % train_acc)
                print("AUC      ----------------- %.4f" % train_auc)
                print("F1 Lower ----------------- %.4f" % train_F1_lower)
                print("F1       ----------------- %.4f" % train_F1)

                with open('loss.txt', 'a') as f:
                    f.write(str(mean_loss) + '\n')
                
                grads = [tsr/i for tsr in grads_sum]
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

                i = 0
                j += 1

        print(f'\n{iterations} iterations, {epoch} epochs completed')

        #####################################
        #####################################
        i = 0
        while i < n_val:
            try:
                pdb_id = next(val_iter)
            except:
                val_iter = iter(val_list)
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

            test_step(X_tf, y_tf)
            i += 1

        val_acc = float(val_acc_metric.result())
        val_auc = float(val_auc_metric.result())
        val_F1_lower = float(val_F1_lower_metric.result())
        val_F1 = float(val_F1_metric.result())

        train_acc_metric.reset_states()
        train_auc_metric.reset_states()
        train_F1_lower_metric.reset_states()
        train_F1_metric.reset_states()

        print(f'\nVALIDATION results over {i} PDBs') 
        print("Accuracy ----------------- %.4f" % val_acc)
        print("AUC      ----------------- %.4f" % val_auc)
        print("F1 Lower ----------------- %.4f" % val_F1_lower)
        print("F1       ----------------- %.4f" % val_F1)

        print(f'\nSaving model weights to {ckpPath}\n')
        model.save_weights(ckpPath)
