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

reg_val = 0.0
reg_type = 'l2'

##########################################
##########################################
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

dev = '/GPU:3'
cpu = '/CPU:0'

params = masif_opts["ligand"]
defaultCode = params['defaultCode']

#datadir = '/data02/daniel/masif/datasets/tf2/masif_ligand'
datadir = '/data02/daniel/masif/datasets/tf2/masif_ligand/allReg'
genPath = os.path.join(datadir, '{}_{}.npy')

train_X = np.load(genPath.format('train', 'X'))
train_y = np.load(genPath.format('train', 'y'))
val_X = np.load(genPath.format('val', 'X'))
val_y = np.load(genPath.format('val', 'y'))

with tf.device(cpu):
  train_X = tf.RaggedTensor.from_tensor(train_X, padding=defaultCode)
  val_X = tf.RaggedTensor.from_tensor(val_X, padding=defaultCode)


modelDir = 'kerasModel'
modelPath = os.path.join(modelDir, 'savedModel')

gpus = tf.config.experimental.list_logical_devices('GPU')
gpus_str = [g.name for g in gpus]
strategy = tf.distribute.MirroredStrategy(gpus_str[1:3])

#with tf.device(dev):
with strategy.scope():
  model = MaSIF_ligand(
    params["max_distance"],
    params["n_classes"],
    feat_mask=params["feat_mask"],
    reg_val = reg_val, reg_type = reg_type,
    learning_rate = lr
  )
  model.compile(optimizer = model.opt,
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
    use_multiprocessing = True
  )

model.save(modelPath)
