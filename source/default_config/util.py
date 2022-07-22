import numpy as np
import tensorflow as tf
from default_config.masif_opts import masif_opts

params = masif_opts["ligand"]
minPockets = params['minPockets']

inputFeatSpec = tf.RaggedTensorSpec(shape=[None, 200, 5], dtype=tf.float32)
restSpec = tf.RaggedTensorSpec(shape=[None, 200, 1], dtype=tf.float32)
sampleSpec = tf.TensorSpec([minPockets], dtype=tf.int32)

inputFeatSpecTsr = tf.TensorSpec(shape=[200, 5], dtype=tf.float32)
restSpecTsr = tf.RaggedTensorSpec(shape=[200, 1], dtype=tf.float32)

prodFunc = lambda a,b : a*b
makeRagged = lambda tsr: tf.RaggedTensor.from_tensor(tsr, ragged_rank = 2)

data_order = ['input_feat', 'rho_coords', 'theta_coords', 'mask']

flatten = lambda a : tf.reshape(a, [-1])

class ValueInit(tf.keras.initializers.Initializer):
  def __init__(self, value):
    self.value = value
  def __call__(self, shape, dtype=None, **kwargs):
    return self.value

def pad_indices(indices, max_verts):
    ret_list = []
    for patch_ix in range(len(indices)):
        ret_list.append(np.concatenate(
            [indices[patch_ix], [patch_ix] * (max_verts - len(indices[patch_ix]))])
        )
    return np.stack(ret_list)

def F1(y_true, y_pred):
    y_true = tf.squeeze(y_true) > 0.0
    y_pred = tf.squeeze(y_pred) > 0.0
    overlap = tf.reduce_sum(tf.cast(y_true & y_pred, dtype=tf.float32))
    n_true = tf.reduce_sum(tf.cast(y_true, dtype=tf.float32))
    n_pred = tf.reduce_sum(tf.cast(y_pred, dtype=tf.float32))
    recall = overlap/n_true
    precision = overlap/n_pred
    return 2*precision*recall / (precision + recall)
