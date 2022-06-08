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
def func(row):
	n_pockets = int(row.shape[0]/8)
	bigShape = [int(n_pockets/200), 200, 5]
	smallShape = [int(n_pockets/200), 200, 1]
	idx = int(functools.reduce(prodFunc, bigShape))
	#print('idx:', idx)
	#print('row.shape[0]:', row.shape[0])
	#print('bigShape:', bigShape)
	#print('smallShape:', smallShape)
	def makeRagged(tsr):
		return tf.RaggedTensor.from_tensor(tsr, ragged_rank = 2)
	feat = makeRagged(tf.reshape(row[:idx], bigShape))
	rest = tf.reshape(row[idx:], [3] + smallShape)
	return [feat, makeRagged(rest[0]), makeRagged(rest[1]), makeRagged(rest[2])]
	#return makeRagged(rest[0])

featType = tf.RaggedTensorSpec(shape=[None, 200, 5], dtype=tf.float32)
restType = tf.RaggedTensorSpec(shape=[None, 200, 1], dtype=tf.float32)
input_feat_full = tf.map_fn(fn=func, elems = test_X, fn_output_signature = [featType, restType, restType, restType])
#input_feat_full = tf.map_fn(fn=func, elems = test_X, fn_output_signature = restType)


