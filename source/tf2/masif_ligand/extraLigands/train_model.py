# Header variables and parameters.
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
from tf2.masif_ligand.MaSIF_ligand_TF2 import MaSIF_ligand

#lr = 1e-3
# Try this learning rate after

reg_val = 0.01
reg_type = 'l2'

continue_training = False
dev = '/GPU:3'
cpu = '/CPU:0'

params = masif_opts["ligand"]
defaultCode = params['defaultCode']

datadir = '/data02/daniel/masif/datasets/tf2/masif_ligand'
genPath = os.path.join(datadir, '{}_{}.npy')

train_X = np.load(genPath.format('train', 'X'), dtype=np.float32)
train_y = np.load(genPath.format('train', 'y'), dtype = np.int32)
val_X = np.load(genPath.format('val', 'X'), dtype=np.float32)
val_y = np.load(genPath.format('val', 'y'), dtype = np.int32)

####
solvents_datadir = '/data02/daniel/masif/datasets/tf2/masif_ligand/extraLigands'
solvents_genPath = os.path.join(solvents_datadir, '{}_{}.npy')

solvents_train_X = np.load(solvents_genPath.format('train', 'X'), dtype=np.float32)
solvents_train_y = np.load(solvents_genPath.format('train', 'y'), dtype = np.int32)
solvents_val_X = np.load(solvents_genPath.format('val', 'X'), dtype=np.float32)
solvents_val_y = np.load(solvents_genPath.format('val', 'y'), dtype = np.int32)

train_y = np.concatenate([train_y, np.zeros([train_y.shape[0], solvents_train_y.shape[1] - train_y.shape[1]])], axis=1)
val_y = np.concatenate([val_y, np.zeros([val_y.shape[0], solvents_train_y.shape[1] - val_y.shape[1]])], axis=1)


####

with tf.device(cpu):
  train_X = tf.RaggedTensor.from_tensor(train_X, padding=defaultCode)
  val_X = tf.RaggedTensor.from_tensor(val_X, padding=defaultCode)
  solvents_train_X = tf.RaggedTensor.from_tensor(solvents_train_X, padding=defaultCode)
  solvents_val_X = tf.RaggedTensor.from_tensor(solvents_val_X, padding=defaultCode)


train_X = tf.concat([train_X, solvents_train_X], axis=0)
train_y = np.concatenate([train_y, solvents_train_y])
val_X = np.concatenate([val_X, solvents_val_X])
val_y = np.concatenate([val_y, solvents_val_y])


modelDir = 'kerasModel'
ckpPath = os.path.join(modelDir, 'ckp')
modelPath = os.path.join(modelDir, 'savedModel')

gpus = tf.config.experimental.list_logical_devices('GPU')
gpus_str = [g.name for g in gpus]
strategy = tf.distribute.MirroredStrategy(gpus_str[2:4])

num_epochs = 200
#with tf.device(dev):
with strategy.scope():
  model = MaSIF_ligand(
    params["max_distance"],
    params["n_classes"],
    feat_mask=params["feat_mask"],
    reg_val = reg_val, reg_type = reg_type,
    keep_prob=0.8
  )
  model.compile(optimizer = model.opt,
    loss = model.loss_fn,
    metrics = ['categorical_accuracy']
  )  
  if continue_training:
    model.load_weights(ckpPath)
    last_epoch = 18
    initValThresh = 0.71429
  else:
    last_epoch = 0
    initValThresh = 0

  saveCheckpoints = tf.keras.callbacks.ModelCheckpoint(
    ckpPath,
    monitor = 'val_categorical_accuracy',
    #save_best_only = True,
    verbose = 1,
    initial_value_threshold = initValThresh
  )

  model.fit(x = train_X, y = train_y,
    epochs = num_epochs,
    initial_epoch = last_epoch,
    validation_data = (val_X, val_y),
    callbacks = [saveCheckpoints],
    verbose = 2,
    use_multiprocessing = True
  )

model.save(modelPath)
