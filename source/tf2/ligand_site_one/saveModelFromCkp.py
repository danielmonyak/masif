import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import sys
import importlib
import numpy as np
import tensorflow as tf
from default_config.util import *
from tensorflow.keras import layers, Sequential, Model
from default_config.util import *

params = masif_opts["ligand"]

modelDir = 'kerasModel'
ckpPath = os.path.join(modelDir, 'ckp')
modelPath = os.path.join(modelDir, 'savedModel')

model = Sequential([
  layers.Dense(1, activation="relu"),
  layers.Flatten(),
  layers.Dense(64, activation="relu"),
  layers.Dense(20, activation='relu'),
  layers.Dense(1)
])

input_shape = [None, 200, 5]
model.build(input_shape)

model.load_weights(ckpPath)
model.save(modelPath)
print('Saved model!')
