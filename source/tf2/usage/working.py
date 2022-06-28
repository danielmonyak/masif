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

modelDir = '../ligand_site/kerasModel'
ckpPath = os.path.join(modelDir, 'ckp')

model = MaSIF_ligand_site(
  params["max_distance"],
  params["n_classes"],
  feat_mask=params["feat_mask"],
  keep_prob = 1.0,
  n_conv_layers = 1
)
model.load_weights(ckpPath)

#pdb = '1RI4_A_'
pdb = '4YTP_ACBD_'

target_pdb = pdb.rstrip('_')
#test_data = tf.data.TFRecordDataset(os.path.join(params["tfrecords_dir"], 'testing_data_sequenceSplit_30.tfrecord')).map(_parse_function)
train_data = tf.data.TFRecordDataset(os.path.join(params["tfrecords_dir"], 'training_data_sequenceSplit_30.tfrecord')).map(_parse_function)
for i, data_element in enumerate(train_data):
  if data_element[5] != target_pdb:
    continue
  
  labels_raw = tf.cast(data_element[4] > 0, dtype=tf.int32)
  labels = tf.squeeze(labels_raw)
  pocket_points = tf.squeeze(tf.where(labels != 0))
  npoints = pocket_points.shape[0]
  savedPockets_temp = min(savedPockets, npoints)
  
  ##
  #pocket_points = tf.random.shuffle(pocket_points)[:savedPockets_temp]
  #npoints = savedPockets_temp
  ##
  pocket_empties = tf.squeeze(tf.where(labels == 0))
  #empties_sample = tf.random.shuffle(pocket_empties)[:npoints]
  empties_sample = pocket_empties
  
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


def map_func(row):
  pocket_points = tf.where(row == 1)
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
  y_pred = tf.math.sigmoid(tf.squeeze(model(X, sample)))
  y_pred = tf.cast(y_pred > 0.5, dtype=tf.int32)
  y_true = tf.squeeze(tf.gather(params = y, indices = sample, axis = 1, batch_dims = 1))

y_true_all = flatten(y_true)
y_pred_all = flatten(y_pred)
bal_acc = balanced_accuracy_score(y_true_all, y_pred_all)
mask = tf.boolean_mask(y_pred_all, y_true_all)
overlap = tf.reduce_sum(mask)
recall = overlap/tf.reduce_sum(y_true_all)
precision = overlap/tf.reduce_sum(y_pred_all)
specificity = 1 - tf.reduce_mean(tf.cast(tf.boolean_mask(y_pred_all, 1 - y_true_all), dtype=tf.float64))

print('Balanced accuracy:', round(bal_acc, 2))
print('Recall:', round(recall.numpy(), 2))
print('Precision:', round(precision.numpy(), 2))
print('Specificity:', round(specificity.numpy(), 2))



precom_dir = '/data02/daniel/masif/data_preparation/04a-precomputation_12A/precomputation'
pdb_dir = os.path.join(precom_dir, pdb)

xyz_coords = Predictor.getXYZCoords(pdb_dir)            
all_ligand_coords = np.load(
    os.path.join(
        ligand_coord_dir, "{}_ligand_coords.npy".format(pdb.split("_")[0])
    )
)
ligand_coords = all_ligand_coords[0]
tree = spatial.KDTree(xyz_coords)
pocket_points_true = tree.query_ball_point(ligand_coords, 3.0)
pocket_points_true = list(set([pp for p in pocket_points_true for pp in p]))


'''
do = lambda x : pred.predictLigandIdx(pred.getLigandX(x))
for thresh in [.5, .6, .7, .8, .9, .95, .99]:
  print('threshold:', thresh)
  pred = Predictor(ligand_model_path, ligand_site_ckp_path, threshold = .5, ligand_threshold = thresh)
  pred.loadData(pdb_dir)
  pocket_points_pred = pred.predictPocketPoints()
  print(do(pocket_points_pred))
'''

ligand_model_path = '/home/daniel.monyak/software/masif/source/tf2/masif_ligand/kerasModel/savedModel'
ligand_site_ckp_path = '/home/daniel.monyak/software/masif/source/tf2/ligand_site/kerasModel/ckp'

pred = Predictor(ligand_model_path, ligand_site_ckp_path)
pred.loadData(pdb_dir)
ligand_site_probs = getLigandSiteProbs()

threshold = 0.5
pocket_points_pred = tf.squeeze(tf.where(ligand_site_preds > threshold))

#####
y_gen = np.zeros(pred.n_pockets)
y_true = y_gen.copy()
y_true[pocket_points_true] = 1
y_pred = y_gen
y_pred[pocket_points_pred.numpy()] = 1


y_true_all = flatten(y_true)
y_pred_all = flatten(y_pred)
bal_acc = balanced_accuracy_score(y_true_all, y_pred_all)
mask = tf.boolean_mask(y_pred_all, y_true_all)
overlap = tf.reduce_sum(mask)
recall = overlap/tf.reduce_sum(y_true_all)
precision = overlap/tf.reduce_sum(y_pred_all)
specificity = 1 - tf.reduce_mean(tf.cast(tf.boolean_mask(y_pred_all, 1 - y_true_all), dtype=tf.float64))

print('Balanced accuracy:', round(bal_acc, 2))
print('Recall:', round(recall.numpy(), 2))
print('Precision:', round(precision.numpy(), 2))
print('Specificity:', round(specificity.numpy(), 2))

#####

X_true = pred.getLigandX(pocket_points_true)
X_true_pred = pred.predictLigandIdx(X_true)
print(X_true_pred)

X_pred = pred.getLigandX(pocket_points_pred)
X_pred_pred = pred.predictLigandIdx(X_pred, 0.5)
print(X_pred_pred)
