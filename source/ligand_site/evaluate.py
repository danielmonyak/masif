# Header variables and parameters.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import numpy as np
from IPython.core.debugger import set_trace
import importlib
import sys
from default_config.masif_opts import masif_opts
from MaSIF_ligand_site import MaSIF_ligand_site
from sklearn.metrics import confusion_matrix
import pickle
import tensorflow as tf

continue_training = False

params = masif_opts["ligand"]
defaultCode = params['defaultCode']

datadir = '/data02/daniel/masif/datasets/ligand_site'
genPath = os.path.join(datadir, '{}_{}.npy')

test_X = np.load(genPath.format('test', 'X'))
test_y = np.load(genPath.format('test', 'y'))

##
def binarize_y(y):
  y[y > 0] = 1
  return tf.constant(y)

test_y = binarize_y(test_y)
##

cpu = '/CPU:0'
with tf.device(cpu):
  test_X = tf.RaggedTensor.from_tensor(test_X, padding=defaultCode)
  test_y = tf.RaggedTensor.from_tensor(test_y, padding=defaultCode)

ligand_site_model = MaSIF_ligand_site(
      params["max_distance"],
      params["n_classes"],
      feat_mask=params["feat_mask"],
      keep_prob = 1.0
    )
    ligand_site_model.compile(optimizer = ligand_site_model.opt,
      loss = ligand_site_model.loss_fn,
      metrics=['accuracy']
    )
    ligand_site_model.load_weights(ligand_site_ckp_path)



modelDir = 'kerasModel'
ckpPath = os.path.join(modelDir, 'ckp')

last_epoch = 0
initValThresh = None

if continue_training:
  model.load_weights(ckpPath)
  last_epoch += 100
  initValThresh = 0.69159


gpus = tf.config.experimental.list_logical_devices('GPU')
gpus_str = [g.name for g in gpus]
strategy = tf.distribute.MirroredStrategy(gpus_str[1:])


saveCheckpoints = tf.keras.callbacks.ModelCheckpoint(
  ckpPath,
  monitor = 'val_binary_accuracy',
  save_best_only = True,
  verbose = 1,
  initial_value_threshold = initValThresh
)

dev = '/GPU:3'

num_epochs = 100
#with strategy.scope():
with tf.device(dev):
  history = model.fit(x = train_X, y = train_y,
    epochs = num_epochs - last_epoch,
    validation_data = (val_X, val_y),
    callbacks = [saveCheckpoints],
    verbose = 2,
    use_multiprocessing = True
  )


'''
with open(os.path.join(modelDir, 'train_history'), 'wb') as file_pi:
  pickle.dump(history.history, file_pi)
'''
