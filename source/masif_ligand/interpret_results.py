import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import numpy as np
from IPython.core.debugger import set_trace
import importlib
import sys
from default_config.masif_opts import masif_opts
from masif_modules.MaSIF_ligand_new import MaSIF_ligand
from masif_modules.read_ligand_tfrecords import _parse_function
from sklearn.metrics import confusion_matrix
import tensorflow as tf

params = masif_opts["ligand"]
test_set_out_dir = params["test_set_out_dir"]

# Load testing data
testing_data = tf.data.TFRecordDataset(
    os.path.join(params["tfrecords_dir"], "testing_data_sequenceSplit_30.tfrecord")
)
testing_data = testing_data.map(_parse_function)

n_ligands = params["n_classes"]
'''
for ligand in range(n_ligands):
  labels = np.loadtxt(test_set_out_dir + "{}_labels.npy".format(prot))
  logits_softmax = np.loadtxt(test_set_out_dir + "{}_logits.npy".format(prot))
'''
num_test_samples = 290
with tf.Session() as sess:
    testing_iterator = testing_data.make_one_shot_iterator()
    testing_next_element = testing_iterator.get_next()
    for num_test_sample in range(num_test_samples):
        print('\nnum_test_sample: ', num_test_sample)
        try:
            data_element = sess.run(testing_next_element)
        except:
            continue

        pdb = data_element[5]
        labels = np.load(test_set_out_dir + "{}_labels.npy".format(pdb)).astype(float)
        print(labels)
    '''
    pdb = '4D86_A'
    labels = np.load(test_set_out_dir + "b'{}'_labels.npy".format(pdb)).astype(float)
    logits_softmax = np.load(test_set_out_dir + "b'{}'_logits.npy".format(prot)).astype(float)
    .reshape([-1, n_ligands])

    print(labels)
    print(logis_softmax)
    '''
    #conf_mat = confusion_matrix(y_true, y_pred)
