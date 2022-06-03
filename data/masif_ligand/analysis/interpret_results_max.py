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

saved_pdbs = np.loadtxt('/home/daniel.monyak/software/masif/data/masif_ligand/saved_pdbs.txt', dtype='str')

'''
# Load testing data
testing_data = tf.data.TFRecordDataset(
    os.path.join(params["tfrecords_dir"], "testing_data_sequenceSplit_30.tfrecord")
)
testing_data = testing_data.map(_parse_function)

testing_iterator = testing_data.make_one_shot_iterator()
testing_next_element = testing_iterator.get_next()
sess = tf.Session()
'''
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
'''
num_test_samples = 290
for num_test_sample in range(num_test_samples):
    try:
        data_element = sess.run(testing_next_element)
    except:
        continue

    pdb = data_element[5]
    labels = data_element[4]
    all_ligands = np.unique(labels.max(axis = 0))
    if len(all_ligands) != 1:
        print('pdb: ', pdb)
        print(all_ligands)
sess.close()
'''

y_true = []
y_pred = []
zero_dim = 0
for pdb in saved_pdbs:
    labels = np.load(test_set_out_dir + "{}_labels.npy".format(pdb)).astype(float)
    if labels.shape[0] == 0:
        zero_dim += 1
        continue
    
    logits_softmax = np.load(test_set_out_dir + "{}_logits.npy".format(pdb)).astype(float)
    y_true.append(labels[0])
    freq_list = []
    for i in range(logits_softmax.shape[0]):
        temp = logits_softmax[i].reshape([-1, n_ligands])
        (unique, counts) = np.unique(temp.argmax(axis = 1), return_counts=True)
        freqs = np.asarray((unique, counts)).T
        freq_list.append(freqs)

    df_list = list(map(lambda freqs : pd.DataFrame(freqs), freq_list))
    total_freqs = pd.concat(df_list).groupby(0).sum()[1]
    y_pred.append(total_freqs.idxmax())
    

#.reshape([-1, n_ligands])
'''conf_mat = confusion_matrix(y_true, y_pred, normalize = 'true')
disp = ConfusionMatrixDisplay(conf_mat)
disp.plot()
plt.savefig('confusion_matrix.png')
'''
print(balanced_accuracy_score(y_true, y_pred))

#sess.close()

