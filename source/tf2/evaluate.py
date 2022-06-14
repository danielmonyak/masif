import tensorflow as tf
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
import numpy as np
import os
from default_config.masif_opts import masif_opts

params = masif_opts["ligand"]
defaultCode = params['defaultCode']


modelDir = 'kerasModel'
#ckpPath = os.path.join(modelDir, 'model')
modelPath = os.path.join(modelDir, 'savedModel')

model = tf.keras.models.load_model(modelPath)

datadir = '/data02/daniel/masif/datasets/tf2'
#train_X = np.load(datadir + 'train_X.npy')
#train_y = np.load(datadir + 'train_y.npy')
#val_X = np.load(datadir + 'val_X.npy')
#val_y = np.load(datadir + 'val_y.npy')
test_X_raw = np.load(os.path.join(datadir, 'test_X.npy'))
test_y = np.load(os.path.join(datadir, 'test_y.npy'))

defaultCode = 123.45679

with tf.device('/CPU:0'):
  test_X = tf.RaggedTensor.from_tensor(test_X_raw, padding=defaultCode)

gpus = tf.config.experimental.list_logical_devices('GPU')
gpus_str = [g.name for g in gpus]
strategy = tf.distribute.MirroredStrategy(gpus_str[1:])

#with strategy.scope():
  #train_res = model.evaluate(train_X, train_y, use_multiprocessing=True)
  #val_res = model.evaluate(val_X, val_y, use_multiprocessing=True)
  #test_res = model.evaluate(test_X, test_y, use_multiprocessing=True)
with tf.device('/GPU:3'):
  y_pred_probs = model.predict(test_X, use_multiprocessing=True)

#print('model.evaluate:', test_res)

y_true = test_y.argmax(axis = 1)
y_pred = y_pred_probs.argmax(axis = 1)

balanced_acc = balanced_accuracy_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_pred_probs, multi_class = 'ovr', labels = np.arange(7))

print('Balanced accuracy:', balanced_acc)
print('ROC AUC:', roc_auc)
