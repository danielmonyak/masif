import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import sys
import importlib
import numpy as np
import tensorflow as tf
from default_config.util import *
from tensorflow.keras import layers, Sequential, Model
from default_config.util import *
from tf2.ligand_site_one.MaSIF_ligand_site_one import MaSIF_ligand_site

params = masif_opts['ligand_site']

modelDir = 'kerasModel'
ckpPath = os.path.join(modelDir, 'ckp')
modelPath = os.path.join(modelDir, 'savedModel')

model = MaSIF_ligand_site(
    params["max_distance"],
    feat_mask=params["feat_mask"],
    n_thetas=4,
    n_rhos=3,
    n_rotations=4,
    learning_rate = 1e-3,
    n_conv_layers = params['n_conv_layers'],
    conv_batch_size = None
)

from_logits = model.loss_fn.get_config()['from_logits']
binAcc = tf.keras.metrics.BinaryAccuracy(threshold = (not from_logits) * 0.5)

model.compile(optimizer = model.opt,
  loss = model.loss_fn,
  metrics=[binAcc]
)

k = 1010
input_feat_empty = np.zeros([1, k, 100, 5], dtype=np.float32)
coords_empty = np.zeros([1, k, 100], dtype=np.float32)
mask_empty = np.zeros([1, k, 100, 1], dtype=np.float32)
indices_empty = np.zeros([1, k, 100], dtype=np.int32)
X_empty = ((input_feat_empty, coords_empty, coords_empty, mask_empty), indices_empty)
_=model(X_empty)

model.load_weights(ckpPath)
model.save(modelPath)
print('Saved model!')
