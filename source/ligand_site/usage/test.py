from predictor import *

ligand_model_path = 'kerasModel/savedModel'
#ligand_site_model_path = '../kerasModel/savedModel'
ligand_site_ckp_path = '../kerasModel/ckp'

pdb = '1C75_A_'
precom_dir = '/data02/daniel/masif/data_preparation/04a-precomputation_12A/precomputation'
pred = Predictor(ligand_model_path, ligand_site_ckp_path)

self= pred
pdb_dir = os.path.join(precom_dir, pdb)
X = pred.getData(pdb_dir)

'''
i = 0
sample = tf.expand_dims(tf.range(minPockets * i, minPockets * (i+1)), axis=0)

a = pred.ligand_site_model(X, sample)
'''
