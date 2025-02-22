import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import sys
import numpy as np
import tensorflow as tf
import pickle

phys_gpus = tf.config.list_physical_devices('GPU')
for phys_g in phys_gpus:
    tf.config.experimental.set_memory_growth(phys_g, True)

import default_config.util as util
from default_config.masif_opts import masif_opts
from tf2.masif_ligand.MaSIF_ligand_TF2 import MaSIF_ligand

params = masif_opts["ligand"]

##########################################
##########################################

# Reularization coefficient
reg_val = 0.0
dev = '/GPU:1'
cpu = '/CPU:0'

# Use batch normalization layers
use_bn = True

##########################################
########################################## Gather input passed collected by input prompts
with open('train_vars.pickle', 'rb') as handle:
    train_vars = pickle.load(handle)

continue_training = train_vars['continue_training']
ckpPath = train_vars['ckpPath']
num_epochs = train_vars['num_epochs']
starting_epoch = train_vars['starting_epoch']
lr = train_vars['lr']

print(f'Training for {num_epochs} epochs')
if continue_training:
    print(f'Resuming training from checkpoint at {ckpPath}, starting at epoch {starting_epoch}, using learning rate {lr:.1e}')

##########################################
##########################################
# Code that pads the arrays
defaultCode = params['defaultCode']

datadir = '/data02/daniel/masif/datasets/tf2/masif_ligand/allReg'
genPath = os.path.join(datadir, '{}_{}.npy')

train_X = np.load(genPath.format('train', 'X'))
train_y = np.load(genPath.format('train', 'y'))
val_X = np.load(genPath.format('val', 'X'))
val_y = np.load(genPath.format('val', 'y'))

# Delete rows with NaN values
valid_train_mask = ~np.isnan(train_X).any(axis=1)
valid_val_mask = ~np.isnan(val_X).any(axis=1)

train_X = train_X[valid_train_mask]
train_y = train_y[valid_train_mask]

val_X = val_X[valid_val_mask]
val_y = val_y[valid_val_mask]

# Convert arrays to ragged tensors
with tf.device(cpu):
  train_X = tf.RaggedTensor.from_tensor(train_X, padding=defaultCode)
  val_X = tf.RaggedTensor.from_tensor(val_X, padding=defaultCode)

modelDir = 'kerasModel'
modelPath = os.path.join(modelDir, 'savedModel')

# Create training strategy with GPUS - can change which GPUS are being used
using_gpus = [1, 2, 3]
gpus = tf.config.experimental.list_logical_devices('GPU')
gpus_str = [g.name for g in gpus]
strategy = tf.distribute.MirroredStrategy([gpus_str[i] for i in using_gpus])

# Can train on one GPU or on multiple, may run out of memory just using one GPU
# Change "use_multiprocessing" variables

#use_multiprocessing = False
#with tf.device(dev):

use_multiprocessing = True
with strategy.scope():
  model = MaSIF_ligand(
    params["max_distance"],
    params["n_classes"],
    feat_mask=params["feat_mask"],
    reg_val = reg_val,
    use_bn = use_bn
  )
  model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=lr),
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics = ['sparse_categorical_accuracy']
  )  
  if continue_training:
    model.load_weights(ckpPath)

  saveCheckpoints = tf.keras.callbacks.ModelCheckpoint(
    ckpPath,
    save_best_only = False,
    verbose = 0
  )

  history = model.fit(x = train_X, y = train_y,
    epochs = num_epochs,
    initial_epoch = starting_epoch,
    validation_data = (val_X, val_y),
    callbacks = [saveCheckpoints],
    verbose = 2,
    use_multiprocessing = use_multiprocessing
  )

model.save(modelPath)
