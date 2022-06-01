# Header variables and parameters.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 


import numpy as np
from IPython.core.debugger import set_trace
import importlib
import sys
from default_config.masif_opts import masif_opts
#####
# Edited by Daniel Monyak
from MaSIF_ligand import MaSIF_ligand
#####
from read_ligand_tfrecords import _parse_function
from sklearn.metrics import confusion_matrix
import tensorflow as tf

continue_training = True


params = masif_opts["ligand"]


# Load dataset
training_data = tf.data.TFRecordDataset(
    os.path.join(params["tfrecords_dir"], "training_data_sequenceSplit_30.tfrecord")
)
validation_data = tf.data.TFRecordDataset(
    os.path.join(params["tfrecords_dir"], "validation_data_sequenceSplit_30.tfrecord")
)
testing_data = tf.data.TFRecordDataset(
    os.path.join(params["tfrecords_dir"], "testing_data_sequenceSplit_30.tfrecord")
)
training_data = training_data.map(_parse_function)
validation_data = validation_data.map(_parse_function)
testing_data = testing_data.map(_parse_function)

training_data = iter(training_data)
validation_data = iter(validation_data)
testing_data = iter(testing_data)



out_dir = params["model_dir"]
output_model = out_dir + "model"
if not os.path.exists(params["model_dir"]):
    os.makedirs(params["model_dir"])

    

# Create Model
model = MaSIF_ligand(
  params["max_distance"],
  params["n_classes"],
  idx_gpu="/gpu:0",
  feat_mask=params["feat_mask"],
  costfun=params["costfun"],
)

model.compile(optimizer = model.opt,
  loss = model.loss_fn,
  metrics=['accuracy']
)

