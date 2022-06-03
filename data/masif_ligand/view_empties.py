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

'''
data_element = sess.run(testing_next_element)

labels = data_element[4]
n_ligands = labels.shape[1]
pdb_logits_softmax = []
pdb_labels = []

print('pdb: ', data_element[5])

print('labels: ', labels)
print('n_ligands: ', n_ligands)

ligand = 0
print('ligand: ', ligand)

pocket_points = np.where(labels[:, ligand] != 0.0)[0]
label = np.max(labels[:, ligand]) - 1

print('label: ', label)

pocket_labels = np.zeros(7, dtype=np.float32)
pocket_labels[label] = 1.0
npoints = pocket_points.shape[0]
print('npoints', npoints)

pdb_labels.append(label)
pdb = data_element[5]
'''
bad_pdbs = ["b'4ZNL_A'", "b'3D36_ACB'", "b'2REQ_AB'", "b'4H6Q_A'", "b'4BLV_A'", "b'1X7P_AB'", "b'5L8J_AB'", "b'5HMN_CE'", "b'3UQD_ACBD'", "b'1XX6_AB'", "b'2OWM_AC'"]
num_test_samples = 290
with tf.device('/GPU:0'):
    for num_test_sample in range(num_test_samples):
        print(num_test_sample)
        try:
            data_element = sess.run(testing_next_element)
        except:
            continue

        pdb = data_element[5]
        print(pdb)
        if pdb in bad_pdbs:
          break
'''
    labels = data_element[4]
    all_ligands = np.unique(labels.max(axis = 0))
    if len(all_ligands) != 1:
        print('pdb: ', pdb)
        print(all_ligands)
sess.close()
'''
