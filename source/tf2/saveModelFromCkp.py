import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

from default_config.masif_opts import masif_opts
from MaSIF_ligand_TF2 import MaSIF_ligand
import tensorflow as tf
import numpy as np

params = masif_opts["ligand"]

modelDir = 'kerasModel'
ckpPath = os.path.join(modelDir, 'ckp')
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

datadir = '/data02/daniel/masif/datasets/tf2'
test_X = np.load(os.path.join(datadir, 'test_X.npy'))
X = test_X[:1]

_ = model(X)

model.load_weights(ckpPath)
model.save(modelPath)
print('Saved model!')
