# Header variables and parameters.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import numpy as np
from IPython.core.debugger import set_trace
import importlib
import sys
from default_config.masif_opts import masif_opts
#####
from prep_data import model
#####
from sklearn.metrics import confusion_matrix
import tensorflow as tf

datadir = 'datasets/'
train_X = np.load(datadir + 'train_X.npy')
train_y = np.load(datadir + 'train_y.npy')
val_X = np.load(datadir + 'val_X.npy')
val_y = np.load(datadir + 'val_X.npy')

num_epochs = 100
#num_batches = 32
model.fit(x = train_X, y = train_y,
          epochs = num_epochs,
          validation_data = (val_X, val_y)
         )
