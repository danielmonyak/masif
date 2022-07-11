import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import sys
import numpy as np
import tensorflow as tf
from default_config.util import *
from tf2.usage.predictor import Predictor

#gpus = tf.config.list_physical_devices('GPU')
#for gpu in gpus:
#    tf.config.experimental.set_memory_growth(gpu, True)

dev = '/GPU:1'
cpu = '/CPU:0'

precom_dir = '/data02/daniel/masif/masif_ligand/data_preparation/04a-precomputation_12A/precomputation'
pdbList = os.listdir(precom_dir)

lists_dir = '/home/daniel.monyak/software/masif/data/masif_ligand'

train_pdbs = np.load(os.path.join(lists_dir, "lists/train_pdbs_sequence.npy")).astype(str)
val_pdbs = np.load(os.path.join(lists_dir, "lists/val_pdbs_sequence.npy")).astype(str)
test_pdbs = np.load(os.path.join(lists_dir, "lists/test_pdbs_sequence.npy")).astype(str)

pdbList_dict = {'train' : train_pdbs, 'val' : val_pdbs, 'test' : test_pdbs}

for dataset in pdbList_dict.keys():
    pdbList = pdbList_dict[dataset]
    n_pdbs = len(pdbList)
    coords_list = []
    for i, pdb_dir in enumerate(pdbList):
        print(f'{dataset} record {i} of {n_pdbs}')
        try:
            coords_list.append(Predictor.getXYZCoords(os.path.join(precom_dir, pdb_dir + '_')))
        except:
            continue

    np.save('train_XYZ
