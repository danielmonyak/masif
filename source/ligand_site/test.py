import os
import tensorflow as tf
from ligand_site.usage.predictor import Predictor
from default_config.util import *

pdb='2VRB_AB_'
#pdb='3AYI_AB_'
#pdb='1C75_A_'


params = masif_opts['ligand']
ligand_coord_dir = params["ligand_coords_dir"]

precom_dir = '/data02/daniel/masif/data_preparation/04a-precomputation_12A/precomputation'
ligand_model_path = '/home/daniel.monyak/software/masif/source/tf2/kerasModel/savedModel'
ligand_site_ckp_path = '/home/daniel.monyak/software/masif/source/ligand_site/kerasModel/ckp'
pred = Predictor(ligand_model_path, ligand_site_ckp_path)
'''
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
