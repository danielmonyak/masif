import tensorflow as tf
import numpy as np
import os

modelDir = 'kerasModel'
ckpPath = os.path.join(modelDir, 'model')
modelPath = os.path.join(modelDir, 'savedModel')

model = tf.keras.models.load_model(modelPath)

datadir = 'datasets/'
train_X = np.load(datadir + 'train_X.npy')
train_y = np.load(datadir + 'train_y.npy')
val_X = np.load(datadir + 'val_X.npy')
val_y = np.load(datadir + 'val_y.npy')
test_X = np.load(datadir + 'test_X.npy')
test_y = np.load(datadir + 'test_y.npy')

gpus = tf.config.experimental.list_logical_devices('GPU')
gpus_str = [g.name for g in gpus]
strategy = tf.distribute.MirroredStrategy(gpus_str[1:])

with strategy.scope():
  #train_res = model.evaluate(train_X, train_y, use_multiprocessing=True)
  #val_res = model.evaluate(val_X, val_y, use_multiprocessing=True)
  #test_res = model.evaluate(test_X, test_y, use_multiprocessing=True)
  y_pred = model.predict(test_X, use_multiprocessing=True)
