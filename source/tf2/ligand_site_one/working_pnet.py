import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import sys
import importlib
import numpy as np
from scipy import spatial
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import tensorflow as tf
from default_config.util import *
from tf2.usage.predictor import Predictor

import math

params = masif_opts["ligand"]
ligand_coord_dir = params["ligand_coords_dir"]
ligand_list = params['ligand_list']

binding_dir = '/data02/daniel/PUresNet/site_predictions'

#pdb = sys.argv[1]
#pdb='7T7A_A'
pdb='5MOG_ACBED_'

print('pdb:', pdb)

precom_dir = params['masif_precomputation_dir']

xyz_coords = Predictor.getXYZCoords(pdb_dir)
tree = spatial.KDTree(xyz_coords)

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

pdb_dir = os.path.join(precom_dir, pdb)
xyz_coords = Predictor.getXYZCoords(pdb_dir)
tree = spatial.KDTree(xyz_coords)
pred.loadData(pdb_dir)

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

#####################
pred = Predictor(ligand_model_path = '/home/daniel.monyak/software/masif/source/tf2/masif_ligand/10/kerasModel/savedModel')
pred.loadData(pdb_dir)

########################
#pdb_pnet_dir = os.path.join(binding_dir, pdb.rstrip("_"))
pdb_pnet_dir ='/home/daniel.monyak/5MOG/charged_5MOG'
files = os.listdir(pdb_pnet_dir)
n_pockets_pred = np.sum(np.char.endswith(files, '.txt'))

pnet_coords = np.loadtxt(os.path.join(pdb_pnet_dir, f'pocket{pocket}.txt'), dtype=float)
pocket_points_pred = tree.query_ball_point(pnet_coords, 3.0)
pocket_points_pred = list(set([pp for p in pocket_points_pred for pp in p]))

npoints = len(pocket_points_pred)
print(f'{npoints} predicted pocket points')

########################
pocket = 0
pocket_points_true = pp_true_list[pocket]

overlap = np.intersect1d(pocket_points_true, pocket_points_pred)
recall = len(overlap)/len(pocket_points_true)
precision = len(overlap)/npoints
print('Recall:', round(recall, 2))
print('Precision:', round(precision, 2))

######################
X_pred = pred.getLigandX(pocket_points_pred)
X_pred_pred = pred.predictLigandIdx(X_pred, 0.5)
print('\nX_pred_pred:', X_pred_pred.numpy())

######################
X_true = pred.getLigandX(pocket_points_true)
X_true_pred = pred.predictLigandIdx(X_true)
print('X_true_pred:', X_true_pred.numpy())

print('\nligandIdx_true:', ligandIdx_true)
