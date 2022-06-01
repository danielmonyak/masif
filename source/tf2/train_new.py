# Header variables and parameters.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 


import numpy as np
from IPython.core.debugger import set_trace
import importlib
import sys
from default_config.masif_opts import masif_opts
#####
# Edited by Daniel Monyak
from MaSIF_ligand import MaSIF_ligand
#####
from read_ligand_tfrecords import _parse_function
from sklearn.metrics import confusion_matrix
import tensorflow as tf

continue_training = True


params = masif_opts["ligand"]

model = MaSIF_ligand(
  params["max_distance"],
  params["n_classes"],
  idx_gpu="/gpu:0",
  feat_mask=params["feat_mask"],
  costfun=params["costfun"],
)

model.compile(optimizer = model.opt,
  loss = model.loss_fn,
  metrics=['accuracy']
)

