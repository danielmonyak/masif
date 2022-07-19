import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import sys
from IPython.core.debugger import set_trace
import importlib
import numpy as np
from scipy import spatial
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import tensorflow as tf

phys_gpus = tf.config.list_physical_devices('GPU')
for phys_g in phys_gpus:
    tf.config.experimental.set_memory_growth(phys_g, True)

from default_config.util import *
from tf2.usage.predictor import Predictor

#params = masif_opts["ligand"]
params = masif_opts["ligand_site"]

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
#pdb = '3WV7_ACB_'
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
    pocket_points_true = np.array(list(set([pp for p in pocket_points_true for pp in p])))
    
    if len(pocket_points_true) == 0:
        print(f'\tLigand has no pocket points...')
        continue
    
    pp_true_list.append(pocket_points_true)


ligand_true = all_ligand_types[0]
ligandIdx_true = ligand_list.index(ligand_true)

pred = Predictor(ligand_model_path = '/home/daniel.monyak/software/masif/source/tf2/masif_ligand/10/kerasModel/savedModel')
pred.loadData(pdb_dir)

predPath = os.path.join(params["out_pred_dir"], f'{pdb}y_pred.npy')
ligand_site_probs = np.load(predPath)

def summary(threshold):
  pocket_points_pred = np.asarray(ligand_site_probs > threshold).nonzero()[0]
  npoints = len(pocket_points_pred)
  if npoints < 2 * minPockets:
    print(f'Only {npoints} pocket points were predicted...\n')
    #return npoints_true
    return
  
  print(f'{npoints} pocket points were predicted...\n')
  X_pred = pred.getLigandX(pocket_points_pred)
  ligand_probs_mean = pred.predictLigandProbs(X_pred, 0.5)
  
  ligandIdx_pred = np.argmax(ligand_probs_mean)
  print('\nligandIdx_pred:', ligandIdx_pred, '\n')
  
  max_prob = np.max(ligand_probs_mean)
  print('max_prob:', round(max_prob, 2))


for threshold in np.linspace(.1, .9, 9):
  print('threshold:', threshold)
  ptsDif = summary(threshold)

pocket_points_pred = np.asarray(ligand_site_probs > threshold_best).nonzero()[0]
