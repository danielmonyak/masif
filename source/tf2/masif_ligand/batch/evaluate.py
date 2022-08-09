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
from tf2.masif_ligand.stochastic.MaSIF_ligand import MaSIF_ligand
from tf2.masif_ligand.stochastic.get_data import get_data
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import mode
import matplotlib.pyplot as plt

params = masif_opts["ligand"]
minPockets = params['minPockets']

ckpPath = 'kerasModel/ckp'
include_solvents = False

if include_solvents:
    ligand_list = masif_opts['all_ligands']
else:
    ligand_list = masif_opts['ligand_list']

model = MaSIF_ligand(
    params["max_distance"],
    len(ligand_list),
    feat_mask=params["feat_mask"],
    reg_val = reg_val, reg_type = reg_type,
    keep_prob=1.0
)
model.load_weights(ckpPath)

test_list = np.load('/home/daniel.monyak/software/masif/data/masif_ligand/newPDBs/lists/test_reg.npy')
gpu = '/GPU:2'

n_pred = 100

y_true = []
probs_list = []
with tf.device(gpu):
  print(f'Making {n_pred} predictions for each {dataset} protein...')
  probs_list = []
  for i, pdb_id in test_list:
    if i % 10 == 0:
      print(i)
      
    data = get_data(pdb_id, include_solvents)
    if data is None:
      continue
      
    X, pocket_points, y = data
    for k, pp in enumerate(pocket_points):
      pp_rand = np.random.choice(pp, minPockets, replace=False)
      X_temp = tuple(tf.constant(arr[:, pp_rand]) for arr in X)
      y_true.append(y[k])
      
      temp_probs_list = []
      for j in range(n_pred):
        temp_probs_list.append(tf.nn.softmax(model.predict(X)))
      probs_list.append(np.stack(temp_probs_list))

probs_tsr = tf.stack(probs_list, axis=-1)

y_pred_probs = tf.reduce_mean(probs_tsr, axis=-1)
y_pred = tf.argmax(y_pred_probs, axis = 1)

#y_true = y.argmax(axis = 1)
y_true = y

balanced_acc = balanced_accuracy_score(y_true, y_pred)
acc = accuracy_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_pred_probs, multi_class = 'ovr', labels = np.arange(7))

print('Balanced accuracy:', balanced_acc)
print('Accuracy: ', acc)
print('ROC AUC:', roc_auc)

conf_mat = confusion_matrix(y_true, y_pred, normalize = 'true')
disp = ConfusionMatrixDisplay(conf_mat, display_labels = masif_opts['ligand_list'])
disp.plot()
plt.savefig('confusion_matrix.png')
