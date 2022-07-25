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

import openbabel
import pybel

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
load_status.expect_partial()

data = get_data(pdb.rstrip('_'), training=False)
if data is None:
    sys.exit('Data couldn\'t be retrieved')

X, y, centroid = data
prot_coords = np.squeeze(X[1])

density = tf.sigmoid(model.predict(X)).numpy()

origin = (centroid - params['max_dist'])
step = np.array([1.0 / params['scale']] * 3)

if len(sys.argv) > 2:
    threshold = float(sys.argv[2])
else:
    threshold = 0.5

min_size=50
path = 'outdir'
file_format = 'mol2'

if not os.path.exists(path):
    os.mkdir(path)

voxel_size = (1 / params['scale']) ** 3
bw = closing((density[0] > threshold).any(axis=-1))
cleared = clear_border(bw)

label_image, num_labels = label(cleared, return_num=True)
for i in range(1, num_labels + 1):
    pocket_idx = (label_image == i)
    pocket_size = pocket_idx.sum() * voxel_size
    if pocket_size < min_size:
        label_image[np.where(pocket_idx)] = 0

pockets = label_image

pocket_label_arr = np.unique(pockets)
i=0
for pocket_label in pocket_label_arr[pocket_label_arr > 0]:
    indices = np.argwhere(pockets == pocket_label).astype('float32')
    indices *= step
    indices += origin
    
    np.savetxt(path+'/pocket'+str(i)+'.txt', indices)
    
    mol=openbabel.OBMol()
    for idx in indices:
        a=mol.NewAtom()
        a.SetVector(float(idx[0]),float(idx[1]),float(idx[2]))
    p_mol=pybel.Molecule(mol)
    p_mol.write(file_format,path+'/pocket'+str(i)+'.'+file_format, overwrite=True)
    i+=1

    
~/software/masif/source/tf2/masif_ligand/l2/kerasModel/savedModel/
