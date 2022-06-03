import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import numpy as np
from default_config.masif_opts import masif_opts
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import matplotlib.pyplot as plt

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
    logits_softmax = logits_softmax.reshape([logits_softmax.shape[0], logits_softmax.shape[1], n_ligands])
    avg_softmax = logits_softmax.mean(axis = 1).mean(axis = 0)
    y_pred_list.append(avg_softmax)

#y_true = np.array(y_true_list).reshape([-1, 1]) 
y_true = np.array(y_true_list)
y_pred_probs = np.vstack(y_pred_list)
y_pred = y_pred_probs.argmax(axis = 1)

'''
enc = OneHotEncoder(categories = [np.arange(n_ligands)])
y_true_one_hot = enc.fit_transform(y_true).toarray()
'''

conf_mat = confusion_matrix(y_true, y_pred, normalize = 'true')
disp = ConfusionMatrixDisplay(conf_mat)
disp.plot()
plt.savefig('confusion_matrix.png')

roc_curve(y_t
print(balanced_accuracy_score(y_true, y_pred))
print(roc_auc_score(y_true, y_pred_probs, multi_class = 'ovr', labels = np.arange(7)))
