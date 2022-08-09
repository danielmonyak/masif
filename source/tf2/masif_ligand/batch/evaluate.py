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
    keep_prob=1.0
)
model.load_weights(ckpPath)

test_list = np.load('/home/daniel.monyak/software/masif/data/masif_ligand/newPDBs/lists/test_reg.npy')
gpu = '/GPU:2'

n_pred = 100

'''
@tf.function(experimental_relax_shapes=True)
def test_step(X, pp):
  pp_rand = tf.random.shuffle(pp)[:minPockets]
  X_temp = tuple(tf.gather(tsr, pp_rand, axis=1) for tsr in X)
  return tf.squeeze(tf.nn.softmax(model(X_temp)))
'''
@tf.function(experimental_relax_shapes=True)
def test_step(X):
  n_samples = tf.shape(X[0])[1]
  samp = tf.random.shuffle(tf.range(n_samples))[:minPockets]
  X_temp = tuple(tf.gather(tsr, samp, axis=1) for tsr in X)
  return tf.squeeze(tf.nn.softmax(model(X_temp)))

y_true = []
probs_list = []
with tf.device(gpu):
  print(f'Making {n_pred} predictions for each protein...')
  for i, pdb_id in enumerate(test_list):
    if i % 10 == 0:
      print(i)
    
    '''
    data = get_data(pdb_id, include_solvents)
    if data is None:
      continue
    X, pocket_points, y = data
    X = tuple(tf.constant(arr) for arr in X)
    for k, pp in enumerate(pocket_points):
      pp = tf.constant(pp)
      y_true.append(y[k])
      temp_probs_list = []
      for j in range(n_pred):
        print(f'j: {j}')
        temp_probs_list.append(test_step(X, pp))
      probs_list.append(np.stack(temp_probs_list))'''
    
    try:
      X = np.load(os.path.join(params['masif_precomputation_dir'], pdb_id, 'X.npy'), allow_pickle=True)
      y = np.load(os.path.join(params['masif_precomputation_dir'], pdb_id, 'y.npy'))
    except:
      continue
    
    for k, y_temp in enumerate(y):
      if y_temp >= 7:
        continue
      y_true.append(y_temp)
      X_temp = X[k]
      X_temp = (tf.expand_dims(X_temp[0], axis=0), tf.constant(X_temp[1]), tf.constant(X_temp[2]), tf.expand_dims(X_temp[3], axis=0))
      temp_probs_list = []
      for j in range(n_pred):
        print(f'j: {j}')
        temp_probs_list.append(test_step(X_temp))
      probs_list.append(np.stack(temp_probs_list))

probs_tsr = np.stack(probs_list)

y_pred_probs = np.mean(probs_tsr, axis=1)
y_pred = np.argmax(y_pred_probs, axis=1)


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
