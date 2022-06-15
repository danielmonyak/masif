from predictor import *

ligand_model_path = 'kerasModel/savedModel'
ligand_site_model_path = '../kerasModel/savedModel'
precom_dir = '/data02/daniel/masif/data_preparation/04a-precomputation_12A/precomputation/3O7W_A_'
pred = Predictor(ligand_model_path, ligand_site_model_path)

X = pred.getData(precom_dir)
i = 0
sample = tf.range(minPockets * i, minPockets * (i+1))

#a = pred.ligand_site_model(X, sample = sample)
a = pred.ligand_site_model.myConvLayer(X, sample)
