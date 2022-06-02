import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import numpy as np
from default_config.masif_opts import masif_opts
from masif_modules.MaSIF_ligand_new import MaSIF_ligand
from sklearn.metrics import confusion_matrix
import tensorflow as tf

params = masif_opts["ligand"]
test_set_out_dir = params["test_set_out_dir"]

n_ligands = params["n_classes"]

for ligand in range(n_ligands):
  labels = np.loadtxt(test_set_out_dir + "{}_labels.npy".format(something))
  logits_softmax = np.loadtxt(test_set_out_dir + "{}_logits.npy".format(something))
  

conf_mat = confusion_matrix(y_true, y_pred)
