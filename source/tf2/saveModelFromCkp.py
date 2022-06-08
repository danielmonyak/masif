import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

from default_config.masif_opts import masif_opts
from MaSIF_ligand_TF2 import MaSIF_ligand
import tensorflow as tf

params = masif_opts["ligand"]

model = MaSIF_ligand(
  params["max_distance"],
  params["n_classes"],
  feat_mask=params["feat_mask"]
)
model.compile(optimizer = model.opt,
  loss = model.loss_fn,
  metrics=['accuracy']
)

modelDir = 'kerasModel'
ckpPath = os.path.join(modelDir, 'model')
modelPath = os.path.join(modelDir, 'savedModel')

model.load_weights(ckpPath)
model.save(modelPath)
