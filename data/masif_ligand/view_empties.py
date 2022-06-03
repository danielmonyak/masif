import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import numpy as np
from IPython.core.debugger import set_trace
import importlib
import sys
from default_config.masif_opts import masif_opts
from masif_modules.MaSIF_ligand_new import MaSIF_ligand
from masif_modules.read_ligand_tfrecords import _parse_function
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, balanced_accuracy_score
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

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


bad_pdbs = ["b'4ZNL_A'", "b'3D36_ACB'", "b'2REQ_AB'", "b'4H6Q_A'", "b'4BLV_A'", "b'1X7P_AB'", "b'5L8J_AB'", "b'5HMN_CE'", "b'3UQD_ACBD'", "b'1XX6_AB'", "b'2OWM_AC'"]
num_test_samples = 290
#splits = [0, int(num_test_samples/4), int(num_test_samples/2), int(3 * num_test_samples/4), num_test_samples]
#devices = ['/GPU:0', '/GPU:1', '/GPU:2', '/GPU:3']
bad_data_elements = []
#for i in range(len(devices)):
#    with tf.device(devices[i]):
gpus = tf.compat.v1.config.experimental.list_logical_devices('GPU')
gpus_str = [g.name for g in gpus]
strategy = tf.distribute.MirroredStrategy(gpus_str)
#print("Using ", devices[i])
with strategy.scope():
    #for num_test_sample in range(splits[i], splits[i+1]):
    print(num_test_sample)
    try:
        data_element = sess.run(testing_next_element)
    except:
        continue
    pdb = data_element[5]
    if pdb in bad_pdbs:
      bad_data_elements.append(data_elements)
'''
    labels = data_element[4]
    all_ligands = np.unique(labels.max(axis = 0))
    if len(all_ligands) != 1:
        print('pdb: ', pdb)
        print(all_ligands)
sess.close()
'''
