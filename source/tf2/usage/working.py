# Header variables and parameters.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import sys
from IPython.core.debugger import set_trace
import importlib
import numpy as np
from scipy import spatial
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import tensorflow as tf
from tf2.read_ligand_tfrecords import _parse_function
from default_config.util import *
from tf2.ligand_site.MaSIF_ligand_site import MaSIF_ligand_site
from tf2.usage.predictor import Predictor


params = masif_opts["ligand"]
defaultCode = params['defaultCode']
minPockets = params['minPockets']
savedPockets = params['savedPockets']
ligand_coord_dir = params["ligand_coords_dir"]
ligand_list = params['ligand_list']

pdb = '1RI4_A_'

target_pdb = pdb.rstrip('_')
test_data = tf.data.TFRecordDataset(os.path.join(params["tfrecords_dir"], 'testing_data_sequenceSplit_30.tfrecord')).map(_parse_function)
for i, data_element in enumerate(test_data):
  print(i)
  if data_element[5] != target_pdb:
    continue
  
  labels_raw = tf.cast(data_element[4] > 0, dtype=tf.int32)
  labels = tf.squeeze(labels_raw)
  pocket_points = tf.squeeze(tf.where(labels != 0))
  npoints = pocket_points.shape[0]
  savedPockets_temp = min(savedPockets, npoints)
  
  ##
  pocket_points = tf.random.shuffle(pocket_points)[:savedPockets_temp]
  npoints = savedPockets_temp
  ##
  pocket_empties = tf.squeeze(tf.where(labels == 0))
  empties_sample = tf.random.shuffle(pocket_empties)[:npoints]
  sample = tf.concat([pocket_points, empties_sample], axis=0)
  
  y = tf.expand_dims(tf.gather(labels, sample), axis=0)
  
  input_feat = tf.gather(data_element[0], sample)
  rho_coords = tf.gather(tf.expand_dims(data_element[1], -1), sample)
  theta_coords = tf.gather(tf.expand_dims(data_element[2], -1), sample)
  mask = tf.gather(data_element[3], sample)
  
  feed_dict = {
      'input_feat' : input_feat,
      'rho_coords' : rho_coords,
      'theta_coords' : theta_coords,
      'mask' : mask
  }
  
  def helperInner(tsr_key):
      tsr = feed_dict[tsr_key]
      return tf.reshape(tsr, [-1])
  
  key_list = ['input_feat', 'rho_coords', 'theta_coords', 'mask']
  flat_list = list(map(helperInner, key_list))
  X = tf.expand_dims(tf.concat(flat_list, axis = 0), axis=0)
  
  break


modelDir = '../ligand_site/kerasModel'
ckpPath = os.path.join(modelDir, 'ckp')

model = MaSIF_ligand_site(
  params["max_distance"],
  params["n_classes"],
  feat_mask=params["feat_mask"],
  keep_prob = 1.0,
  n_conv_layers = 4
)
model.load_weights(ckpPath)

def map_func(row):
  pocket_points = tf.where(row != 0)
  pocket_points = tf.random.shuffle(pocket_points)[:int(minPockets/2)]
  pocket_empties = tf.where(row == 0)
  pocket_empties = tf.random.shuffle(pocket_empties)[:int(minPockets/2)]
  return tf.cast(
    tf.squeeze(
      tf.concat([pocket_points, pocket_empties], axis = 0)
    ),
    dtype=tf.int32
  )

sample = tf.map_fn(fn=map_func, elems = y, fn_output_signature = sampleSpec)

dev = '/GPU:3'
with tf.device(dev):
  y_pred = tf.squeeze(model(X, sample))
  y_pred = tf.cast(y_pred > 0.5, dtype=tf.int32)
  y_true = tf.squeeze(tf.gather(params = y, indices = sample, axis = 1, batch_dims = 1))

acc = balanced_accuracy_score(flatten(y_true), flatten(y_pred))
print('Balanced accuracy: ', round(acc, 2))






precom_dir = '/data02/daniel/masif/data_preparation/04a-precomputation_12A/precomputation'
pdb_dir = os.path.join(precom_dir, pdb)

ligand_model_path = '/home/daniel.monyak/software/masif/source/tf2/masif_ligand/kerasModel/savedModel'
ligand_site_ckp_path = '/home/daniel.monyak/software/masif/source/tf2/ligand_site/kerasModel/ckp'

thresh = 0.5
pred = Predictor(ligand_model_path, ligand_site_ckp_path, threshold = thresh)

xyz_coords = pred.getXYZCoords(pdb_dir)            
all_ligand_coords = np.load(
    os.path.join(
        ligand_coord_dir, "{}_ligand_coords.npy".format(pdb.split("_")[0])
    )
)
ligand_coords = all_ligand_coords[0]
tree = spatial.KDTree(xyz_coords)
pocket_points_true = tree.query_ball_point(ligand_coords, 3.0)
pocket_points_true = list(set([pp for p in pocket_points_true for pp in p]))




do = lambda x : pred.predictLigandIdx(pred.getLigandX(x))
for thresh in [.5, .6, .7, .8, .9, .95, .99]:
  print('threshold:', thresh)
  pred = Predictor(ligand_model_path, ligand_site_ckp_path, threshold = .5, ligand_threshold = thresh)
  pred.loadData(pdb_dir)
  pocket_points_pred = pred.predictPocketPoints()
  print(do(pocket_points_pred))

 

'''
#####
y_gen = np.zeros(pred.n_pockets)
y_true = y_gen.copy()
y_true[pocket_points_true] = 1
y_pred = y_gen
y_pred[pocket_points_pred] = 1

acc = balanced_accuracy_score(flatten(y_true), flatten(y_pred))
print('Balanced accuracy: ', round(acc, 2))
#####
'''
'''
X_true = pred.getLigandX(pocket_points_true)
X_true_pred = pred.predictLigandIdx(X_true)

X_pred = pred.getLigandX(pocket_points_pred)
X_pred_pred = pred.predictLigandIdx(X_pred)

'''

'''

def temp_fn(key):
  data = pred.data_dict[key]
  return data.flatten()

flat_list = list(map(temp_fn, data_order))
X = tf.expand_dims(tf.concat(flat_list, axis=0), axis=0)

y_afc = np.zeros([1, pred.n_pockets])
y_afc[0, pocket_points_pred] = 1

sample = tf.map_fn(fn=map_func, elems = y_afc, fn_output_signature = sampleSpec)

dev = '/GPU:3'
with tf.device(dev):
  y_pred = tf.squeeze(model(X, sample))
  y_pred = tf.cast(y_pred > 0.5, dtype=tf.int32)
  y_true = tf.squeeze(tf.gather(params = y_afc, indices = sample, axis = 1, batch_dims = 1))

acc = balanced_accuracy_score(flatten(y_true), flatten(y_pred))
print('Balanced accuracy: ', round(acc, 2))
'''
