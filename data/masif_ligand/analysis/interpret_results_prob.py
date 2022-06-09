import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import numpy as np
from default_config.masif_opts import masif_opts
import tensorflow as tf

params = masif_opts["ligand"]
test_set_out_dir = "/home/daniel.monyak/software/masif/data/masif_ligand/" + params["test_set_out_dir"]
n_ligands = params["n_classes"]

saved_pdbs = np.loadtxt('/home/daniel.monyak/software/masif/data/masif_ligand/saved_pdbs.txt', dtype='str')

y_true_list = []
y_pred_list = []
zero_dim = 0

for pdb in saved_pdbs:
    labels = np.load(test_set_out_dir + "{}_labels.npy".format(pdb)).astype(float)
    if labels.shape[0] == 0:
        zero_dim += 1
        continue
    
    y_true_list.append(labels[0])
    
    logits_softmax = np.load(test_set_out_dir + "{}_logits.npy".format(pdb)).astype(float)
    logits_softmax = np.squeeze(logits_softmax, axis = (2,3))
    avg_softmax = np.mean(logits_softmax, axis = (0,1))
    
    y_pred_list.append(avg_softmax)

y_true = np.array(y_true_list)
y_pred_probs = np.vstack(y_pred_list)
y_pred = y_pred_probs.argmax(axis = 1)

np.savetxt('y_true.txt', y_true)
np.savetxt('y_pred_probs.txt', y_pred_probs)
np.savetxt('y_pred.txt', y_pred)
