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

'''
#pdb = '1RI4_A_' # 0.1 but not correct
#pdb = '1FCD_AC_' # 0.25
#pdb = '2VRB_AB_' # 0.25
possible_test_pdbs = ['2VRB_AB_', '1FCD_AC_', '1FNN_A_', '1RI4_A_', '4PGH_AB_']
possible_train_pdbs = ['3O7W_A_', '4YTP_ACBD_', '4YMP_A_', '4IVM_B_', '3FMO_AB_']
pos_list = {'test' : possible_test_pdbs, 'train' : possible_train_pdbs}

if len(sys.argv) > 2:
  pdb_idx = int(sys.argv[1])
  possible_pdbs = pos_list[sys.argv[2]]
else:
  pdb_idx = 0
  possible_pdbs = possible_test_pdbs

pdb = possible_pdbs[pdb_idx]'''

pdb = sys.argv[1]

print('pdb:', pdb)

'''
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

target_pdb = pdb.rstrip('_')
#test_data = tf.data.TFRecordDataset(os.path.join(params["tfrecords_dir"], 'testing_data_sequenceSplit_30.tfrecord')).map(_parse_function)
train_data = tf.data.TFRecordDataset(os.path.join(params["tfrecords_dir"], 'training_data_sequenceSplit_30.tfrecord')).map(_parse_function)

def goodLabel(labels):
  n_ligands = labels.shape[1]
  if n_ligands > 1:
    return False
  
  pocket_points = tf.where(labels != 0)
  npoints = tf.shape(pocket_points)[0]
  if npoints < minPockets:
    return False
  
  return True

with tf.device('/GPU:3'):
  for i, data_element in enumerate(train_data):
    print(i)
    if data_element[5] != target_pdb:
      continue
    
    labels = data_element[4]
    if not goodLabel(labels):
        continue
    
    y = tf.transpose(tf.cast(labels > 0, dtype=tf.int32))
    flat_list = list(map(flatten, data_element[:4]))
    X = tf.expand_dims(tf.concat(flat_list, axis=0), axis=0)
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

'''

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


all_ligand_types = np.load(
    os.path.join(
        ligand_coord_dir, "{}_ligand_types.npy".format(pdb.split("_")[0])
    )
).astype(str)
ligand_true = all_ligand_types[0]
ligandIdx_true = ligand_list.index(ligand_true)


ligand_model_path = '/home/daniel.monyak/software/masif/source/tf2/masif_ligand/kerasModel/savedModel'
ligand_site_ckp_path = '/home/daniel.monyak/software/masif/source/tf2/ligand_site/kerasModel/ckp'
#ligand_site_ckp_path = '/home/daniel.monyak/software/masif/source/tf2/usage/kerasModel_ligand_site/ckp'

pred = Predictor(ligand_model_path, ligand_site_ckp_path)
pred.loadData(pdb_dir)
ligand_site_probs = pred.getLigandSiteProbs()

def summary(threshold):
  pocket_points_pred = tf.squeeze(tf.where(ligand_site_probs > threshold))
  
  npoints = len(pocket_points_pred)
  if npoints < minPockets:
    print(f'Less than {minPockets} pocket points were predicted...\n')
    return 0, 0
  elif npoints < 2 * minPockets:
    print(f'Less than {2 * minPockets} pocket points were predicted...\n')
    return 0, 0
  
  overlap = np.intersect1d(pocket_points_true, pocket_points_pred)
  recall = len(overlap)/len(pocket_points_true)
  precision = len(overlap)/len(pocket_points_pred)
  print('Recall:', round(recall, 2))
  print('Precision:', round(precision, 2))
  
  #f1 = precision*recall/(precision+recall)
  X_pred = pred.getLigandX(pocket_points_pred)
  ligand_probs_mean = pred.predictLigandProbs(X_pred)
  max_prob = tf.reduce_max(ligand_probs_mean)
  print('\nmax_prob:', round(max_prob.numpy(), 2))
  
  score = max_prob/(1 + abs(.5 - threshold))
  print('score:', round(score.numpy(), 2), '\n')
  
  return max_prob, score


########
print()

#max_prob_best = 0.5
score_best = 0
threshold_best = 0
for threshold in np.linspace(.1, .9, 9):
  print('threshold:', threshold)
  max_prob, score = summary(threshold)
  if max_prob > 0.5 and score > score_best:
    #max_prob_best = max_prob
    score_best = score
    threshold_best = threshold

print('threshold_best:', threshold_best)
print()
if not threshold_best:
  threshold_best = 0.5
  print('NO THRESHOLD WAS GOOD ENUGH TO GIVE A PREDICTION WITH CONFIDENCE')

pocket_points_pred = tf.squeeze(tf.where(ligand_site_probs > threshold_best))
########
'''
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

print()
'''
########

X_true = pred.getLigandX(pocket_points_true)
X_true_pred = pred.predictLigandIdx(X_true)
print('X_true_pred:', X_true_pred.numpy())

X_pred = pred.getLigandX(pocket_points_pred)
X_pred_pred = pred.predictLigandIdx(X_pred, 0.5)
print('\nX_pred_pred:', X_pred_pred.numpy())

print('\nligandIdx_true:', ligandIdx_true)
