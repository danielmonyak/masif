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

import math

params = masif_opts["ligand"]
ligand_coord_dir = params["ligand_coords_dir"]
ligand_list = params['ligand_list']


pdb = sys.argv[1]

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

npoints_true = len(pocket_points_true)
print(f'{npoints_true} true pocket points')

all_ligand_types = np.load(
    os.path.join(
        ligand_coord_dir, "{}_ligand_types.npy".format(pdb.split("_")[0])
    )
).astype(str)
ligand_true = all_ligand_types[0]
ligandIdx_true = ligand_list.index(ligand_true)


ligand_model_path = '/home/daniel.monyak/software/masif/source/tf2/usage/masif_ligand_model/savedModel'
ligand_site_model_path = '/home/daniel.monyak/software/masif/source/tf2/ligand_site_one/kerasModel/savedModel'

pred = Predictor(ligand_model_path = ligand_model_path, ligand_site_model_path = ligand_site_model_path)
pred.loadData(pdb_dir)

########################
pnet_coords = np.loadtxt(f'/home/daniel.monyak/software/PUResNet/output_folders/{pdb.rstrip("_")}/pocket0.txt', dtype=float)
pocket_points_pred = tree.query_ball_point(pnet_coords, 3.0)
pocket_points_pred = list(set([pp for p in pocket_points_pred for pp in p]))

'''
cos_part = np.identity(2)
sin_part = np.array([[0, -1], [1, 0]])
for theta in np.linspace(0, 2*math.pi, 100):
    print(f'theta: {theta}')
    rot_mat = math.cos(theta) * cos_part + math.sin(theta) * sin_part
    coords_new = np.concatenate([np.matmul(rot_mat, pnet_coords[:, :2].T).T, np.expand_dims(pnet_coords[:, 2], axis=-1)], axis=1)
    
    pocket_points_pred = tree.query_ball_point(coords_new, 3.0)
    pocket_points_pred = list(set([pp for p in pocket_points_pred for pp in p]))
    
    npoints = len(pocket_points_pred)
    
    overlap = np.intersect1d(pocket_points_true, pocket_points_pred)
    if len(overlap) > 0:
        recall = len(overlap)/npoints_true
        precision = len(overlap)/npoints
        print('Recall:', round(recall, 2))
        print('Precision:', round(precision, 2), '\n')
'''
########################

npoints = len(pocket_points_pred)
print(f'{npoints} predicted pocket points')

overlap = np.intersect1d(pocket_points_true, pocket_points_pred)
recall = len(overlap)/npoints_true
precision = len(overlap)/npoints
print('Recall:', round(recall, 2))
print('Precision:', round(precision, 2))



X_true = pred.getLigandX(pocket_points_true)
X_true_pred = pred.predictLigandIdx(X_true)
print('X_true_pred:', X_true_pred.numpy())

X_pred = pred.getLigandX(pocket_points_pred)
X_pred_pred = pred.predictLigandIdx(X_pred, 0.5)
print('\nX_pred_pred:', X_pred_pred.numpy())

print('\nligandIdx_true:', ligandIdx_true)
