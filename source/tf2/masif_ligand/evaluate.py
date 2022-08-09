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
from tf2.masif_ligand.MaSIF_ligand_TF2 import MaSIF_ligand
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import mode
import matplotlib.pyplot as plt

params = masif_opts["ligand"]
defaultCode = params['defaultCode']

modelDir = 'kerasModel'
if len(sys.argv) > 1:
    basedir = sys.argv[1]
    modelDir = os.path.join(basedir, modelDir)
modelPath = os.path.join(modelDir, 'savedModel')
model = tf.keras.models.load_model(modelPath)

print(f'Loaded model from {modelPath}')

datadir = '/data02/daniel/masif/datasets/tf2/masif_ligand/allReg'
genPath = os.path.join(datadir, '{}_{}.npy')

if len(sys.argv) > 2:
    dataset = sys.argv[2]
else:
    dataset = 'test'


X = np.load(genPath.format(dataset, 'X'))
y = np.load(genPath.format(dataset, 'y'))

defaultCode = params['defaultCode']
gpu = '/GPU:2'
cpu = '/CPU:0'

n_pred = 100

with tf.device(cpu):
  X = tf.RaggedTensor.from_tensor(X, padding=defaultCode)

with tf.device(gpu):
  print(f'Making {n_pred} predictions for each {dataset} protein...')
  probs_list = []
  for i in range(n_pred):
    if i % 10 == 0:
      print(i)
    probs_list.append(tf.nn.softmax(model.predict(X)))

probs_tsr = tf.stack(probs_list, axis=-1)

'''
preds_tsr = tf.argmax(probs_tsr, axis=1)
y_pred = []
for i in range(len(preds_tsr)):
  y_pred.append(mode(preds_tsr[i].numpy()).mode)

'''

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
