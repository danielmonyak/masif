import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import sys
import pickle
import numpy as np
import tensorflow as tf
from default_config.util import *
from tf2.read_ligand_tfrecords import _parse_function
from tf2.LSResNet.LSResNet import LSResNet
from tf2.usage.predictor import Predictor

import tfbio.data
from LSResNet import runLayers

precom_dir = '/data02/daniel/masif/masif_ligand/data_preparation/04a-precomputation_12A/precomputation'

params = masif_opts["ligand"]

dataset_list = {'train' : "training_data_sequenceSplit_30.tfrecord", 'val' : "validation_data_sequenceSplit_30.tfrecord", 'test' : "testing_data_sequenceSplit_30.tfrecord"}
getData = lambda dataset : tf.data.TFRecordDataset(os.path.join(params["tfrecords_dir"], dataset_list[dataset])).map(_parse_function)
train_data = getData('train')



train_iterator = iter(train_data)
data_element = train_iterator.get_next()

labels = data_element[4]


pdb = data_element[5].numpy().decode('ascii') + '_'
pdb_dir = os.path.join(precom_dir, pdb)
xyz_coords = tf.cast(tf.expand_dims(Predictor.getXYZCoords(pdb_dir), axis=0), dtype=tf.float32)

coords = [tf.expand_dims(tsr, axis=-1) for tsr in data_element[1:3]]
X = tf.expand_dims(tf.concat([data_element[0]] + coords + [data_element[3]], axis=-1), axis=0)

y_raw = tf.cast(labels > 0, dtype=tf.int32)
resolution = 1. / .5
y = tfbio.data.make_grid(xyz_coords[0], y_raw, max_dist=35, grid_resolution=resolution)

X_packed = (X, xyz_coords)
dev = '/GPU:2'
################

X_packed = (X[:, :100], xyz_coords[:, :100])
#y = tf.expand_dims(y_raw[:100], axis=0)

from time import process_time


gpus = tf.config.experimental.list_logical_devices('GPU')
gpus_str = [g.name for g in gpus]
strategy = tf.distribute.MirroredStrategy([gpus_str[0], gpus_str[1]])

before = process_time()
#with tf.device(dev):
with strategy.scope():
    #outputs = model(X_packed)
    model = LSResNet(
        params["max_distance"],
        params["n_classes"],
        feat_mask=params["feat_mask"]
    )
    
    from_logits = model.loss_fn.get_config()['from_logits']
    binAcc = tf.keras.metrics.BinaryAccuracy(threshold = (not from_logits) * 0.5)
    
    model.compile(optimizer = model.opt,
      loss = model.loss_fn,
      metrics=[binAcc]
    )
    
    model.fit(X_packed, y, epochs=1, steps_per_epoch=1, batch_size=1, use_multiprocessing = False)

after = process_time()
print(f'Took {round(after-before)} seconds')

