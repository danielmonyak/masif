import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

from default_config.masif_opts import masif_opts
from MaSIF_ligand_TF2 import MaSIF_ligand
import tensorflow as tf
import numpy as np

params = masif_opts["ligand"]

modelDir = 'kerasModel'
ckpPath = os.path.join(modelDir, 'model')
modelPath = os.path.join(modelDir, 'savedModel')

model = MaSIF_ligand(
  params["max_distance"],
  params["n_classes"],
  feat_mask=params["feat_mask"]
)
model.compile(optimizer = model.opt,
  loss = model.loss_fn,
  metrics=['accuracy']
)

import numpy as np
datadir = 'datasets/'
train_X = np.load(datadir + 'train_X.npy')
train_y = np.load(datadir + 'train_y.npy')
_ = model.predict(train_X[:1])

#model.build([None, model.bigLen + model.smallLen * 3])
'''
model.load_weights(ckpPath)
model.save(modelPath)
'''
