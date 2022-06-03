import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import numpy as np
from IPython.core.debugger import set_trace
import importlib
import sys
from default_config.masif_opts import masif_opts
from masif_modules.read_ligand_tfrecords import _parse_function
import tensorflow as tf

params = masif_opts["ligand"]
test_set_out_dir = params["test_set_out_dir"]
n_ligands = params["n_classes"]

# Load testing data
testing_data = tf.data.TFRecordDataset(
    os.path.join(params["tfrecords_dir"], "testing_data_sequenceSplit_30.tfrecord")
)
testing_data = testing_data.map(_parse_function)

testing_iterator = testing_data.make_one_shot_iterator()
testing_next_element = testing_iterator.get_next()
sess = tf.Session()

data_element = sess.run(testing_next_element)

target_pdb = "b'1U5U_A'"

num_test_samples = 290
gpus = tf.compat.v1.config.experimental.list_logical_devices('GPU')
gpus_str = [g.name for g in gpus]
strategy = tf.distribute.MirroredStrategy(gpus_str)
with strategy.scope():
    for num_test_sample in range(num_test_samples):
        print(num_test_sample)
        try:
            data_element = sess.run(testing_next_element)
        except:
            continue

        pdb = data_element[5]
        if pdb != target_pdb:
            continue
        labels = data_element[4]
        print(labels)
        break
sess.close()
