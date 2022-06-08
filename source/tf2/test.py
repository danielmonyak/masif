import tensorflow as tf
import functools
import numpy as np
from default_config.masif_opts import masif_opts

params = masif_opts["ligand"]
defaultCode = params['defaultCode']
minPockets = 32

datadir = 'datasets/'
test_X_raw = np.load(datadir + 'test_X.npy')
print(test_X_raw[0].shape[0])


test_X = tf.RaggedTensor.from_tensor(test_X_raw, padding=defaultCode)
print(test_X[0].shape[0])

prodFunc = lambda a,b : a*b
makeRagged = lambda tsr: tf.RaggedTensor.from_tensor(tsr, ragged_rank = 2)
def func(row):
	n_pockets = int(row.shape[0]/8)
	bigShape = [int(n_pockets/200), 200, 5]
	smallShape = [int(n_pockets/200), 200, 1]
	idx = int(functools.reduce(prodFunc, bigShape))
	input_feat = tf.reshape(row[:idx], bigShape)
	rest = tf.reshape(row[idx:], [3] + smallShape)
	sample = np.random.choice(n_pockets, minPockets, replace = False)
	data_list = [makeRagged(tsr) for tsr in [input_feat, rest[0], rest[1], rest[2]]]
	return [data_list, sample]

inputFeatType = tf.RaggedTensorSpec(shape=[None, 200, 5], dtype=tf.float32)
restType = tf.RaggedTensorSpec(shape=[None, 200, 1], dtype=tf.float32)
ret = tf.map_fn(fn=func, elems = test_X, fn_output_signature = [inputFeatType, restType, restType, restType])

data_list = ret[0]
sample = ret[1]
inputFeatType, rho_coords, theta_coords, mask = [tf.gather(data, sample, axis = 1, batch_dims = 1) for data in data_list]
