import tensorflow as tf
from default_config.masif_opts import masif_opts

params = masif_opts["ligand"]
minPockets = params['minPockets']

inputFeatSpec = tf.RaggedTensorSpec(shape=[None, 200, 5], dtype=tf.float32)
restSpec = tf.RaggedTensorSpec(shape=[None, 200, 1], dtype=tf.float32)
sampleSpec = tf.TensorSpec([minPockets], dtype=tf.int32)

prodFunc = lambda a,b : a*b
makeRagged = lambda tsr: tf.RaggedTensor.from_tensor(tsr, ragged_rank = 2)

data_order = ['input_feat', 'rho_coords', 'theta_coords', 'mask']

flatten = lambda a : tf.reshape(a, [-1])

class ValueInit(tf.keras.initializers.Initializer):
  def __init__(self, value):
    self.value = value
  def __call__(self, shape, dtype=None, **kwargs):
    return self.value

def printd(var):
  print(var + ':', eval(var))
