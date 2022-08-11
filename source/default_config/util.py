# Useful constants, functions, and classes

import numpy as np
import tensorflow as tf
from tensorflow.keras import metrics
from default_config.masif_opts import masif_opts

minPockets = masif_opts['ligand']['minPockets']

# Used in MaSIF_ligand model
inputFeatSpec = tf.RaggedTensorSpec(shape=[None, 200, 5], dtype=tf.float32)
restSpec = tf.RaggedTensorSpec(shape=[None, 200, 1], dtype=tf.float32)
sampleSpec = tf.TensorSpec([minPockets], dtype=tf.int32)
inputFeatSpecTsr = tf.TensorSpec(shape=[200, 5], dtype=tf.float32)
restSpecTsr = tf.RaggedTensorSpec(shape=[200, 1], dtype=tf.float32)

prodFunc = lambda a,b : a*b
makeRagged = lambda tsr: tf.RaggedTensor.from_tensor(tsr, ragged_rank = 2)

data_order = ['input_feat', 'rho_coords', 'theta_coords', 'mask']

flatten = lambda a : tf.reshape(a, [-1])

# Custom intializer to use in "self.add_weight"
class ValueInit(tf.keras.initializers.Initializer):
  def __init__(self, value):
    self.value = value
  def __call__(self, shape, dtype=None, **kwargs):
    return self.value

# Pad indices array with value of index of that row
def pad_indices(indices, max_verts):
    ret_list = []
    for patch_ix in range(len(indices)):
        ret_list.append(np.concatenate(
            [indices[patch_ix], [patch_ix] * (max_verts - len(indices[patch_ix]))])
        )
    return np.stack(ret_list)

  
def runLayers(layers, x):
    for l in layers:
        x = l(x)
    return x

'''
def F1(y_true, y_pred, threshold=0.5):
    y_true = tf.cast(y_true, dtype=tf.bool)
    y_pred = tf.sigmoid(y_pred) > threshold
    overlap = tf.reduce_sum(tf.cast(y_true & y_pred, dtype=tf.float32))
    n_true = tf.reduce_sum(tf.cast(y_true, dtype=tf.float32))
    n_pred = tf.reduce_sum(tf.cast(y_pred, dtype=tf.float32))
    recall = overlap/n_true
    precision = overlap/n_pred
    f1 = 2*precision*recall / (precision + recall)
    return tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
'''

# Custom Keras metric to calculate F1 score
class F1_Metric(metrics.Metric):
    def __init__(self, name=None, from_logits=False, threshold=0.5, **kwargs):
        if name is None:
            name = f'F1_{threshold}'
        super(F1_Metric, self).__init__(name=name, **kwargs)
        self.f1_score = self.add_weight(name='f1_score', initializer='zeros')
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
        self.from_logits = from_logits
        self.threshold = threshold
    def update_state(self, y_true, y_pred, **kwargs):
        y_true = tf.cast(y_true, dtype=tf.float32)
        if self.from_logits:
            y_pred = tf.sigmoid(y_pred)
        y_true = tf.reshape(y_true, [-1]) > self.threshold
        y_pred = tf.reshape(y_pred, [-1]) > self.threshold
        
        overlap = tf.reduce_sum(tf.cast(y_true & y_pred, dtype=tf.float32))
        n_true = tf.reduce_sum(tf.cast(y_true, dtype=tf.float32))
        n_pred = tf.reduce_sum(tf.cast(y_pred, dtype=tf.float32))
        recall = overlap/n_true
        precision = overlap/n_pred
        f1 = 2*precision*recall / (precision + recall)
        f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
        self.total.assign_add(tf.cast(f1, self.dtype))
        self.count.assign_add(tf.cast(1, self.dtype))
    def result(self):
        return self.total/self.count

# Custom Keras metric to calculate accuracy when using the hinge-loss model
class HingeAccuracy(metrics.Metric):
    def __init__(self, name='hinge_accuracy', **kwargs):
        super(HingeAccuracy, self).__init__(name=name, **kwargs)
        self.hinge_acc_score = self.add_weight(name='hinge_acc_score', initializer='zeros')
        self.n_matching = self.add_weight(name='n_matching', initializer='zeros')
        self.n_total = self.add_weight(name='n_total', initializer='zeros')
    def update_state(self, y_true, y_pred):
        y_true = tf.reshape(y_true, [-1]) > 0.0
        y_pred = tf.reshape(y_pred, [-1]) > 0.0
        result = tf.cast(y_true == y_pred, tf.float32)
        n_matching = tf.reduce_sum(result)
        n_total = tf.shape(result)[0]
        self.n_matching.assign_add(tf.cast(n_matching, self.dtype))
        self.n_total.assign_add(tf.cast(n_total, self.dtype))
    def result(self):
        self.hinge_acc_score = self.n_matching/self.n_total
        return self.hinge_acc_score

# Custom Keras metric to calculate F1 score when using the hinge-loss model
class F1_Metric_Hinge(metrics.Metric):
    def __init__(self, name='F1', **kwargs):
        super(F1_Metric, self).__init__(name=name, **kwargs)
        self.f1_score = self.add_weight(name='f1_score', initializer='zeros')
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
    def update_state(self, y_true, y_pred):
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_true = tf.reshape(y_true, [-1]) > 0.0
        y_pred = tf.reshape(y_pred, [-1]) > 0.0
        overlap = tf.reduce_sum(tf.cast(y_true & y_pred, dtype=tf.float32))
        n_true = tf.reduce_sum(tf.cast(y_true, dtype=tf.float32))
        n_pred = tf.reduce_sum(tf.cast(y_pred, dtype=tf.float32))
        recall = overlap/n_true
        precision = overlap/n_pred
        f1 = 2*precision*recall / (precision + recall)
        f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
        self.total.assign_add(tf.cast(f1, self.dtype))
        self.count.assign_add(tf.cast(1, self.dtype))
    def result(self):
        return self.total/self.count
