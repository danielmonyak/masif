import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import sys
from IPython.core.debugger import set_trace
import importlib
import numpy as np
from scipy import spatial
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import tensorflow as tf

tf.config.set_soft_device_placement(True)

phys_gpus = tf.config.list_physical_devices('GPU')
for phys_g in phys_gpus:
    tf.config.experimental.set_memory_growth(phys_g, True)

from default_config.util import *
from tf2.usage.predictor import Predictor

params = masif_opts["ligand"]
ligand_coord_dir = params["ligand_coords_dir"]
ligand_list = params['ligand_list']

'''possible_test_pdbs = ['2VRB_AB_', '1FCD_AC_', '1FNN_A_', '1RI4_A_', '4PGH_AB_']
possible_train_pdbs = ['3O7W_A_', '4YTP_ACBD_', '4YMP_A_', '4IVM_B_', '3FMO_AB_']
pos_list = {'test' : possible_test_pdbs, 'train' : possible_train_pdbs}

if len(sys.argv) > 2:
  pdb_idx = int(sys.argv[1])
  possible_pdbs = pos_list[sys.argv[2]]
else:
  pdb_idx = 0
  possible_pdbs = possible_test_pdbs

pdb = possible_pdbs[pdb_idx]
'''
#pdb = sys.argv[1]
pdb = '3O7W_A_'

print('pdb:', pdb)

pdb_dir = os.path.join(params['masif_precomputation_dir'], pdb)

####################
all_ligand_types = np.load(
    os.path.join(
        ligand_coord_dir, "{}_ligand_types.npy".format(pdb.split("_")[0])
    )
).astype(str)
all_ligand_coords = np.load(
    os.path.join(
        ligand_coord_dir, "{}_ligand_coords.npy".format(pdb.split("_")[0])
    )
)
n_pockets_true = len(all_ligand_types)

xyz_coords = Predictor.getXYZCoords(pdb_dir)
tree = spatial.KDTree(xyz_coords)

pp_true_list = []
for lig_i in range(n_pockets_true):
    print(f'Pocket {lig_i}')
    
    ligand_coords = all_ligand_coords[lig_i]
    pocket_points_true = tree.query_ball_point(ligand_coords, 3.0)
    pocket_points_true = list(set([pp for p in pocket_points_true for pp in p]))
    
    if len(pocket_points_true) == 0:
        print(f'\tLigand has no pocket points...')
        continue
    
    pp_true_list.append(pocket_points_true)


ligand_true = all_ligand_types[0]
ligandIdx_true = ligand_list.index(ligand_true)

ligand_model_path = '/home/daniel.monyak/software/masif/source/tf2/masif_ligand/10/kerasModel/savedModel'
ligand_site_model_path = '/home/daniel.monyak/software/masif/source/tf2/ligand_site_one/kerasModel/savedModel'

#with tf.device(dev):
pred = Predictor(ligand_model_path = ligand_model_path, ligand_site_model_path = ligand_site_model_path)
pred.loadData(pdb_dir)
X = ((pred.input_feat, pred.rho_coords, pred.theta_coords, pred.mask), pred.indices)
ligand_site_probs = tf.sigmoid(pred.ligand_site_model(X))

def summary(threshold):
  pocket_points_pred = flatten(tf.where(tf.squeeze(ligand_site_probs > threshold)))
  
  npoints = len(pocket_points_pred)
  if npoints < 2 * minPockets:
    print(f'Only {npoints} pocket points were predicted...\n')
    return npoints_true
  
  print(f'{npoints} pocket points were predicted...\n')
  '''
  overlap = np.intersect1d(pocket_points_true, pocket_points_pred)
  recall = len(overlap)/len(pocket_points_true)
  precision = len(overlap)/npoints
  print('Recall:', round(recall, 2))
  print('Precision:', round(precision, 2))
  '''
  X_pred = pred.getLigandX(pocket_points_pred)
  ligand_probs_mean = pred.predictLigandProbs(X_pred, 0.5)
  
  ligandIdx_pred = tf.argmax(ligand_probs_mean)
  print('\nligandIdx_pred:', ligandIdx_pred.numpy(), '\n')
  
  max_prob = tf.reduce_max(ligand_probs_mean)
  print('max_prob:', round(max_prob.numpy(), 2))
  #
  #score = max_prob/(1 + abs(.5 - threshold))
  #print('score:', round(score.numpy(), 2), '\n')
  #return max_prob, score
  
  return abs(npoints - npoints_true)


########
print()

#score_best = 0
ptsDif_best = npoints_true
threshold_best = 0
for threshold in np.linspace(.1, .9, 9):
  print('threshold:', threshold)
  
  ptsDif = summary(threshold)
  if ptsDif < ptsDif_best:
      ptsDif_best = ptsDif
      threshold_best = threshold
  #max_prob, score = summary(threshold)
  #if max_prob > 0.5 and score > score_best:
  #  score_best = score
  #  threshold_best = threshold

print('threshold_best:', threshold_best)
print()
if not threshold_best:
  threshold_best = 0.5
  sys.exit('NO THRESHOLD WAS GOOD ENOUGH TO GIVE A PREDICTION WITH CONFIDENCE')

pocket_points_pred = np.asarray(ligand_site_probs > threshold_best).nonzero()[0]
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

