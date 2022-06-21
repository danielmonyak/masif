import os
import tensorflow as tf
from ligand_site.usage.predictor import Predictor

pdb='2VRB_AB_'
#pdb='3AYI_AB_'
#pdb='1C75_A_'

precom_dir = '/data02/daniel/masif/data_preparation/04a-precomputation_12A/precomputation'
ligand_model_path = '/home/daniel.monyak/software/masif/source/tf2/kerasModel/savedModel'
ligand_site_ckp_path = '/home/daniel.monyak/software/masif/source/ligand_site/kerasModel/ckp'
pred = Predictor(ligand_model_path, ligand_site_ckp_path)

pdb_dir = os.path.join(precom_dir, pdb)
pred.loadData(pdb_dir)
pocket_points = pred.predictPocketPoints()


