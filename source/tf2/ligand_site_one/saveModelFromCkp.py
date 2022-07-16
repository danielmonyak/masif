import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import sys
import importlib
import numpy as np
import tensorflow as tf
from default_config.util import *
from tensorflow.keras import layers, Sequential, Model
from default_config.util import *
from MaSIF_ligand_site_one import MaSIF_ligand_site

params = masif_opts["ligand"]

modelDir = 'kerasModel'
ckpPath = os.path.join(modelDir, 'ckp')
modelPath = os.path.join(modelDir, 'savedModel')

model = MaSIF_ligand_site(
    params["max_distance"],
    params["n_classes"],
    feat_mask=params["feat_mask"],
    n_conv_layers = masif_opts['site']['n_conv_layers'],
    conv_batch_size = 1000
)
from_logits = model.loss_fn.get_config()['from_logits']
binAcc = tf.keras.metrics.BinaryAccuracy(threshold = (not from_logits) * 0.5)
model.compile(optimizer = model.opt,
  loss = model.loss_fn,
  metrics=[binAcc]
)

k = 1010
input_feat_empty = tf.zeros([k, 200, 5])
coords_empty = tf.zeros([k, 200])
mask_empty = tf.zeros([k, 200, 1])
X_empty = ((input_feat_empty, coords_empty, coords_empty, mask_empty), coords_empty)
_=model(X_empty)

model.load_weights(ckpPath)
model.save(modelPath)
print('Saved model!')
