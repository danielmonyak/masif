import tensorflow as tf
import functools
import numpy as np
from default_config.masif_opts import masif_opts

params = masif_opts["ligand"]
defaultCode = params['defaultCode']

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
	feat = tf.reshape(row[:idx], bigShape)
	rest = tf.reshape(row[idx:], [3] + smallShape)
	return [makeRagged(tsr) for tsr in [feat, rest[0], rest[1], rest[2]]

featType = tf.RaggedTensorSpec(shape=[None, 200, 5], dtype=tf.float32)
restType = tf.RaggedTensorSpec(shape=[None, 200, 1], dtype=tf.float32)
data_list = tf.map_fn(fn=func, elems = test_X, fn_output_signature = [featType, restType, restType, restType])

