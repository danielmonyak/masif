import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import numpy as np

from default_config.masif_opts import masif_opts
from MaSIF_ligand_site import MaSIF_ligand_site
import tensorflow as tf
import numpy as np

params = masif_opts["ligand"]

modelDir = 'kerasModel'
ckpPath = os.path.join(modelDir, 'ckp')
modelPath = os.path.join(modelDir, 'savedModel')

model = MaSIF_ligand_site(
  params["max_distance"],
  params["n_classes"],
  feat_mask=params["feat_mask"],
  keep_prob = 1.0
)
model.compile(optimizer = model.opt,
  loss = model.loss_fn,
  metrics=['accuracy']
)

datadir = '/data02/daniel/masif/datasets/ligand_site'
X = np.load(os.path.join(datadir, 'test_X.npy'))
_ = model.predict(X[:1])

model.load_weights(ckpPath)
model.save(modelPath)
