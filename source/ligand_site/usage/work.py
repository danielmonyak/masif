import numpy as np
import os
from default_config.util import *
from ligand_site.MaSIF_ligand_site import MaSIF_ligand_site
from time import process_time

params = masif_opts['ligand']
ligand_list = params['ligand_list']
minPockets = params['minPockets']


pdb = '1C75_A_'
#pdb='3AYI_AB_'

precom_dir = '/data02/daniel/masif/data_preparation/04a-precomputation_12A/precomputation'
ligand_model_path = '/home/daniel.monyak/software/masif/source/tf2/kerasModel/savedModel'
ligand_site_ckp_path = '/home/daniel.monyak/software/masif/source/ligand_site/kerasModel/ckp'

pred = Predictor(ligand_model_path, ligand_site_ckp_path)
pdb_dir = os.path.join(precom_dir, pdb)

pred.loadData(pdb_dir)
ligand_site_X = pred.getLigandSiteX()

gen_sample = tf.expand_dims(tf.range(minPockets), axis = 0)

print(process_time())

i = 3
sample = tf.expand_dims(tf.range(minPockets * i, minPockets * (i+1)), axis = 0)
temp_pred = tf.squeeze(self.ligand_site_model(X, sample))

print(process_time())

def getFlatDataFromDict(key):
  data = self.data_dict[key]
  return data[].flatten() ################################################################# fix
flat_list = list(map(getFlatDataFromDict, data_order))
return tf.RaggedTensor.from_tensor(
  tf.expand_dims(
    tf.concat(flat_list, axis=0),
    axis=0),
  ragged_rank = 1
)

print(process_time())
