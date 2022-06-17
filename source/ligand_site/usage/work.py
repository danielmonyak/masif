import numpy as np
import os
from default_config.util import *
from predictor import Predictor
from time import process_time

params = masif_opts['ligand']
ligand_list = params['ligand_list']
minPockets = params['minPockets']


#pdb = '1C75_A_'
pdb='3AYI_AB_'

precom_dir = '/data02/daniel/masif/data_preparation/04a-precomputation_12A/precomputation'
ligand_model_path = '/home/daniel.monyak/software/masif/source/tf2/kerasModel/savedModel'
ligand_site_ckp_path = '/home/daniel.monyak/software/masif/source/ligand_site/kerasModel/ckp'

pred = Predictor(ligand_model_path, ligand_site_ckp_path)
pdb_dir = os.path.join(precom_dir, pdb)

pred.loadData(pdb_dir)
ligand_site_X = pred.getLigandSiteX()

gen_sample = tf.expand_dims(tf.range(minPockets), axis = 0)
def getFlatDataFromDict(key, j):
  data = pred.data_dict[key]
  return data[minPockets * j : minPockets * (j+1)].flatten()


i = 4

def proc1(i):
  sample = tf.expand_dims(tf.range(minPockets * i, minPockets * (i+1)), axis = 0)
  return tf.squeeze(pred.ligand_site_model(ligand_site_X, sample))

def proc2(i):
  fn = lambda key : getFlatDataFromDict(key, i)
  flat_list = list(map(fn, data_order))
  #temp_X = tf.RaggedTensor.from_tensor(tf.expand_dims(tf.concat(flat_list, axis=0), axis=0), ragged_rank = 1)
  temp_X = tf.expand_dims(tf.concat(flat_list, axis=0), axis=0)
  return tf.squeeze(pred.ligand_site_model(temp_X, gen_sample))

a = process_time()

temp_pred_a = proc1(i)

b = process_time()

temp_pred_b = proc2(i)

c = process_time()

print('First:', b-a)
print('Second:', c-b)
