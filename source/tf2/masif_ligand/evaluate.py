# Header variables and parameters.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf

phys_gpus = tf.config.list_physical_devices('GPU')
for phys_g in phys_gpus:
    tf.config.experimental.set_memory_growth(phys_g, True)

import numpy as np
from IPython.core.debugger import set_trace
import importlib
import sys

import default_config.util as util
from default_config.masif_opts import masif_opts
from tf2.masif_ligand.MaSIF_ligand_TF2 import MaSIF_ligand
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from scipy.stats import mode

params = masif_opts["ligand"]
defaultCode = params['defaultCode']


modelDir = 'kerasModel'
if len(sys.argv) > 1:
    basedir = sys.argv[1]
    modelDir = os.path.join(basedir, modelDir)
modelPath = os.path.join(modelDir, 'savedModel')
model = tf.keras.models.load_model(modelPath)

print(f'Loaded model from {modelPath}')

datadir = '/data02/daniel/masif/datasets/tf2/masif_ligand'
genPath = os.path.join(datadir, '{}_{}.npy')

'''train_X = np.load(genPath.format('train', 'X'))
train_y = np.load(genPath.format('train', 'y'))'''
val_X = np.load(genPath.format('val', 'X'))
val_y = np.load(genPath.format('val', 'y'))
'''test_X = np.load(genPath.format('test', 'X'))
test_y = np.load(genPath.format('test', 'y'))'''

defaultCode = params['defaultCode']
gpu = '/GPU:3'
cpu = '/CPU:0'

n_pred = 100

with tf.device(cpu):
  #train_X = tf.RaggedTensor.from_tensor(train_X, padding=defaultCode)
  val_X = tf.RaggedTensor.from_tensor(val_X, padding=defaultCode)
  #test_X = tf.RaggedTensor.from_tensor(test_X, padding=defaultCode)


with tf.device(gpu):
  '''print('train')
  train_res = model.evaluate(train_X, train_y, verbose=2)
  print('val')
  val_res = model.evaluate(val_X, val_y, verbose=2)'''
  print(f'Making {n_pred} predictions for each val protein...')
  probs_list = []
  for i in range(n_pred):
    print(i)
    #test_res = model.evaluate(test_X, test_y, verbose=2)
    y_pred_probs_temp = tf.nn.softmax(model.predict(val_X))
    probs_list.append(y_pred_probs_temp)

probs_tsr = tf.stack(probs_list, axis=-1)
'''
preds_tsr = tf.argmax(probs_tsr, axis=1)
y_pred = []
for i in range(len(preds_tsr)):
  y_pred.append(mode(preds_tsr[i].numpy()).mode)

'''
y_pred_probs = tf.reduce_mean(probs_tsr, axis=-1)
y_pred = tf.argmax(y_pred_probs, axis = 1)


y_true = val_y.argmax(axis = 1)

balanced_acc = balanced_accuracy_score(y_true, y_pred)
acc = accuracy_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_pred_probs, multi_class = 'ovr', labels = np.arange(7))

print('Balanced accuracy:', balanced_acc)
print('Accuracy: ', acc)
print('ROC AUC:', roc_auc)
