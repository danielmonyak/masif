import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import sys
import importlib
from IPython.core.debugger import set_trace
import pickle
import numpy as np
from scipy import spatial
import tensorflow as tf

from skimage.segmentation import clear_border
from skimage.morphology import closing
from skimage.measure import label

phys_gpus = tf.config.list_physical_devices('GPU')
for phys_g in phys_gpus:
    tf.config.experimental.set_memory_growth(phys_g, True)

from default_config.util import *
from tf2.LSResNet.LSResNet import LSResNet
from get_data import get_data

params = masif_opts["LSResNet"]
ligand_coord_dir = params["ligand_coords_dir"]
ligand_list = params['ligand_list']

possible_test_pdbs = ['2VRB_AB_', '1FCD_AC_', '1FNN_A_', '1RI4_A_', '4PGH_AB_']
possible_train_pdbs = ['4X7G_A_', '4RLR_A_', '3OWC_A_', '3SC6_A_', '1TU9_A_']
pos_list = {'test' : possible_test_pdbs, 'train' : possible_train_pdbs}

if len(sys.argv) > 2:
  pdb_idx = int(sys.argv[1])
  possible_pdbs = pos_list[sys.argv[2]]
  pdb = possible_pdbs[pdb_idx]
else:
  pdb = sys.argv[1]

print('pdb:', pdb)

model = LSResNet(
    params["max_distance"],
    feat_mask=params["feat_mask"],
    n_thetas=4,
    n_rhos=3,
    learning_rate = 1e-4,
    n_rotations=4,
    reg_val = 0
)
from_logits = model.loss_fn.get_config()['from_logits']
thresh = (not from_logits) * 0.5
binAcc = tf.keras.metrics.BinaryAccuracy(threshold = thresh)
auc = tf.keras.metrics.AUC(from_logits = from_logits)
model.compile(optimizer = model.opt,
  loss = model.loss_fn,
  metrics=[binAcc, auc]
)


modelDir = 'kerasModel'
ckpPath = os.path.join(modelDir, 'ckp')

load_status = model.load_weights(ckpPath)
#load_status.expect_partial()

X, y = get_data(pdb.rstrip('_'))
logits = model.predict(X)
probs = tf.sigmoid(logits).numpy()
'''
voxel_size = (1 / params['scale']) ** 3
bw = closing((density[0] > threshold).any(axis=-1))
cleared = clear_border(bw)

label_image, num_labels = label(cleared, return_num=True)
for i in range(1, num_labels + 1):
    pocket_idx = (label_image == i)
    pocket_size = pocket_idx.sum() * voxel_size
    if pocket_size < min_size:
        label_image[np.where(pocket_idx)] = 0
'''
