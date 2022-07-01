import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import sys
from IPython.core.debugger import set_trace
import importlib
import numpy as np
from scipy import spatial
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import tensorflow as tf
from default_config.util import *
from tf2.usage.predictor import Predictor


params = masif_opts["ligand"]
ligand_coord_dir = params["ligand_coords_dir"]
ligand_list = params['ligand_list']

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

pdb = possible_pdbs[pdb_idx]

#pdb = sys.argv[1]
#pdb = '3O7W_A_'

print('pdb:', pdb)


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


#ligand_model_path = '/home/daniel.monyak/software/masif/source/tf2/masif_ligand/kerasModel/savedModel'
#ligand_model_path = '/home/daniel.monyak/software/masif/source/tf2/usage/masif_ligand_model/savedModel'

ligand_site_model_path = '/home/daniel.monyak/software/masif/source/tf2/ligand_site_one/kerasModel/savedModel'

ligand_site_model = tf.keras.models.load_model(ligand_site_model_path)
  
input_feat = np.load(os.path.join(pdb_dir, "p1_input_feat.npy"))
rho_coords = np.load(os.path.join(pdb_dir, "p1_rho_wrt_center.npy"))
theta_coords = np.load(os.path.join(pdb_dir, "p1_theta_wrt_center.npy"))
mask = np.expand_dims(np.load(os.path.join(pdb_dir, "p1_mask.npy")), axis=-1)

X = (input_feat, rho_coords, theta_coords, mask)
ligand_site_probs = tf.math.sigmoid(ligand_site_model.predict(X))
pocket_points_pred = tf.squeeze(tf.where(tf.squeeze(ligand_site_probs > .5)))

overlap = np.intersect1d(pocket_points_true, pocket_points_pred)
recall = len(overlap)/len(pocket_points_true)
precision = len(overlap)/len(pocket_points_pred)

print('Recall:', round(recall, 2))
print('Precision:', round(precision, 2))


def summary(threshold):
  pocket_points_pred = tf.squeeze(tf.where(tf.squeeze(ligand_site_probs > threshold)))
  
  npoints = len(pocket_points_pred)
  if len(pocket_points_pred) == 0:
    print('No pocket points were predicted...\n')
    return
  print(f'{npoints} pocket points were predicted...')
  
  overlap = np.intersect1d(pocket_points_true, pocket_points_pred)
  recall = len(overlap)/len(pocket_points_true)
  precision = len(overlap)/len(pocket_points_pred)
  print('Recall:', round(recall, 2))
  print('Precision:', round(precision, 2), '\n')


########
print()

#max_prob_best = 0.5
for threshold in np.linspace(.1, .9, 9):
  print('threshold:', threshold)
  summary(threshold)

'''
print('threshold_best:', threshold_best)
print()
if not threshold_best:
  threshold_best = 0.5
  print('NO THRESHOLD WAS GOOD ENUGH TO GIVE A PREDICTION WITH CONFIDENCE')

pocket_points_pred = tf.squeeze(tf.where(ligand_site_probs > threshold_best))
'''
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
'''
X_true = pred.getLigandX(pocket_points_true)
X_true_pred = pred.predictLigandIdx(X_true)
print('X_true_pred:', X_true_pred.numpy())

X_pred = pred.getLigandX(pocket_points_pred)
X_pred_pred = pred.predictLigandIdx(X_pred, 0.5)
print('\nX_pred_pred:', X_pred_pred.numpy())

print('\nligandIdx_true:', ligandIdx_true)

'''
