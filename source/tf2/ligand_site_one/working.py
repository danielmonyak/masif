import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import sys
import numpy as np
from scipy import spatial
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, RocCurveDisplay
import tensorflow as tf
import matplotlib.pyplot as plt

phys_gpus = tf.config.list_physical_devices('GPU')
for phys_g in phys_gpus:
    tf.config.experimental.set_memory_growth(phys_g, True)

from default_config.util import *
from tf2.usage.predictor import Predictor

#params = masif_opts["ligand"]
params = masif_opts["ligand_site"]
ligand_coord_dir = params["ligand_coords_dir"]
ligand_list = params['ligand_list']

possible_test_pdbs = ['2VRB_AB_', '1FCD_AC_', '1FNN_A_', '1RI4_A_', '4PGH_AB_']
#possible_train_pdbs = ['3O7W_A_', '4YTP_ACBD_', '4YMP_A_', '4IVM_B_', '3FMO_AB_']
possible_train_pdbs = ['4X7G_A_', '4RLR_A_', '3OWC_A_', '3SC6_A_', '1TU9_A_']
pos_list = {'test' : possible_test_pdbs, 'train' : possible_train_pdbs}

if len(sys.argv) > 2:
  pdb_idx = int(sys.argv[1])
  possible_pdbs = pos_list[sys.argv[2]]
  pdb = possible_pdbs[pdb_idx]
else:
  pdb = sys.argv[1]

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
    pocket_points_true = np.array(list(set([pp for p in pocket_points_true for pp in p])))
    
    if len(pocket_points_true) == 0:
        print(f'\tLigand has no pocket points...')
        continue
    
    pp_true_list.append(pocket_points_true)

pocket_points_true_all = np.concatenate(pp_true_list)
npoints_true = len(pocket_points_true_all)

ligand_true = all_ligand_types[0]
ligandIdx_true = ligand_list.index(ligand_true)

pred = Predictor(
    ligand_model_path = '/home/daniel.monyak/software/masif/source/tf2/masif_ligand/10/kerasModel/savedModel',
    ligand_site_ckp_path = '/home/daniel.monyak/software/masif/source/tf2/ligand_site_one/sep_layers/kerasModel/ckp'
)
pred.loadData(pdb_dir)

data_tsrs = tuple(np.expand_dims(tsr, axis=0) for tsr in [pred.input_feat, pred.rho_coords, pred.theta_coords, pred.mask])
indices = np.expand_dims(pred.indices, axis=0)
X = (data_tsrs, indices)
logits = pred.ligand_site_model(X)
ligand_site_probs = np.squeeze(tf.sigmoid(logits))

###########################################################################
###########################################################################

def summary(threshold, pocket_points_true_all, score):
    pocket_points_pred = np.asarray(score > threshold).nonzero()[0]  
    npoints = len(pocket_points_pred)
        
    if npoints < 2 * minPockets:
        print(f'Only {npoints} pocket points were predicted...\n')
        return npoints_true
    
    print(f'{npoints} pocket points were predicted...')
    
    overlap = np.intersect1d(pocket_points_true_all, pocket_points_pred)
    recall = len(overlap)/len(pocket_points_true_all)
    precision = len(overlap)/npoints
    print('Recall:', round(recall, 2))
    print('Precision:', round(precision, 2))
    '''
    X_pred = pred.getLigandX(pocket_points_pred)
    ligand_probs_mean = pred.predictLigandProbs(X_pred, 0)
    
    ligandIdx_pred = tf.argmax(ligand_probs_mean)
    print('\nligandIdx_pred:', ligandIdx_pred.numpy(), '\n')
    
    max_prob = tf.reduce_max(ligand_probs_mean)
    print('max_prob:', round(max_prob.numpy(), 2))
    #
    #score = max_prob/(1 + abs(.5 - threshold))
    #print('score:', round(score.numpy(), 2), '\n')
    #return max_prob, score'''
    
    return abs(npoints - npoints_true)

###########################################################################
###########################################################################

print()

#score_best = 0
ptsDif_best = npoints_true
threshold_best = 0
thresh_list = np.concatenate([np.linspace(.1, .9, 9), np.linspace(.91, .99, 9)])
for threshold in thresh_list:
  print('\nthreshold:', threshold)
  
  ptsDif = summary(threshold, pocket_points_true_all, ligand_site_probs)
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
    print('NO THRESHOLD WAS GOOD ENOUGH TO GIVE A PREDICTION WITH CONFIDENCE')

###########################################################################
###########################################################################

pocket_points_pred = np.asarray(ligand_site_probs > threshold_best).nonzero()[0]
########

y_gen = np.zeros(pred.n_samples)
y_true = y_gen.copy()
y_true[pocket_points_true_all] = 1
'''
y_pred = y_gen
y_pred[pocket_points_pred] = 1

bal_acc = balanced_accuracy_score(y_true, y_pred)
mask = (y_pred == 1) & (y_true == 1)
overlap = np.sum(mask)
recall = overlap/np.sum(y_true)
precision = overlap/np.sum(y_pred)
specificity = 1 - np.mean(y_pred == (1 - y_true))

print('Balanced accuracy:', round(bal_acc, 2))
print('Recall:', round(recall, 2))
print('Precision:', round(precision, 2))
print('Specificity:', round(specificity, 2))

print()

########

X_true = pred.getLigandX(pocket_points_true)
X_true_pred = pred.predictLigandIdx(X_true)
print('X_true_pred:', X_true_pred.numpy())

X_pred = pred.getLigandX(pocket_points_pred)
X_pred_pred = pred.predictLigandIdx(X_pred, 0.5)
print('\nX_pred_pred:', X_pred_pred.numpy())

print('\nligandIdx_true:', ligandIdx_true)

'''
print('auc:', roc_auc_score(y_true, ligand_site_probs))
#RocCurveDisplay.from_predictions(y_true, ligand_site_probs)
#plt.savefig('curve')
'''
print('\ntf1\n')

tf1_score = np.load(os.path.join('../../ligand_site', params["out_pred_dir"], f'{pdb}y_pred.npy'))
for threshold in thresh_list:
  print('\nthreshold:', threshold)
  ptsDif = summary(threshold, pocket_points_true_all, tf1_score)

print('auc:', roc_auc_score(y_true, tf1_score))
'''
