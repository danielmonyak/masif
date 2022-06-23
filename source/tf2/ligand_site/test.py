import os
import numpy as np
from scipy import spatial
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import tensorflow as tf
from tf2.usage.predictor import Predictor
from default_config.util import *

#pdb='2VRB_AB_'
pdb = '1RI4_A_'

params = masif_opts['ligand']
ligand_coord_dir = params["ligand_coords_dir"]
ligand_list = params['ligand_list']

precom_dir = '/data02/daniel/masif/data_preparation/04a-precomputation_12A/precomputation'
ligand_model_path = '/home/daniel.monyak/software/masif/source/tf2/masif_ligand/kerasModel/savedModel'
ligand_site_ckp_path = '/home/daniel.monyak/software/masif/source/tf2/ligand_site/kerasModel/ckp'
thresh = 0.5
pred = Predictor(ligand_model_path, ligand_site_ckp_path, threshold = thresh)

pdb_dir = os.path.join(precom_dir, pdb)
pred.loadData(pdb_dir)
pocket_points_pred = pred.predictPocketPoints()

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
'''
y_gen = np.zeros(pred.n_pockets)
y_true = y_gen.copy()
y_true[pocket_points_true] = 1
y_pred = y_gen
y_pred[pocket_points_pred] = 1

acc = accuracy_score(y_true, y_pred)
print('Accuracy: ', round(acc, 2))
'''

X_true = pred.getLigandX(pocket_points_true)
X_true_pred = pred.predictLigandIdx(X_true)

X_pred = pred.getLigandX(pocket_points_pred)
X_pred_pred = pred.predictLigandIdx(X_pred)

t = pocket_points_true
p = pocket_points_pred

do = lambda x : pred.predictLigandIdx(pred.getLigandX(x))
