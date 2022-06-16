from predictor import *

ligand_model_path = '/home/daniel.monyak/software/masif/source/tf2/kerasModel/savedModel'
#ligand_site_model_path = '/software/masif/source/ligand_site/kerasModel/savedModel'
ligand_site_ckp_path = '/home/daniel.monyak/software/masif/source/ligand_site/kerasModel/ckp'

pdb = '1C75_A_'
precom_dir = '/data02/daniel/masif/data_preparation/04a-precomputation_12A/precomputation'
pred = Predictor(ligand_model_path, ligand_site_ckp_path)

pdb_dir = os.path.join(precom_dir, pdb)

ligand_pred, coord_list = pred.predict(pdb_dir)
