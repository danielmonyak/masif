import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import sys
import numpy as np
import tensorflow as tf

phys_gpus = tf.config.list_physical_devices('GPU')
for phys_g in phys_gpus:
    tf.config.experimental.set_memory_growth(phys_g, True)


from default_config.util import *
from tf2.LSResNet.LSResNet import LSResNet
from tf2.LSResNet.predict import predict

params = masif_opts["LSResNet"]
ligand_coord_dir = params["ligand_coords_dir"]

'''
possible_test_pdbs = ['2VRB_AB_', '1FCD_AC_', '1FNN_A_', '1RI4_A_', '4PGH_AB_']
possible_train_pdbs = ['4X7G_A_', '4RLR_A_', '3OWC_A_', '3SC6_A_', '1TU9_A_']
pos_list = {'test' : possible_test_pdbs, 'train' : possible_train_pdbs}
'''

model = LSResNet(
    params["max_distance"],
    feat_mask=params["feat_mask"],
    n_thetas=4,
    n_rhos=3,
    n_rotations=4
)

modelDir = 'kerasModel'
ckpPath = os.path.join(modelDir, 'ckp')
load_status = model.load_weights(ckpPath)
load_status.expect_partial()

########################
pdb = input(f'Enter pdb: ')
if pdb == '':
    sys.exit('Must enter a valid pdb...')

threshold = 0.5
threshold_key = input(f'Enter threshold [{threshold}]: ')
if threshold_key != '':
    try:
        threshold = float(threshold_key)
        if (threshold <= 0) or (threshold >= 1):
            sys.exit('Must be a number between 0 and 1 (exclusive)...')
    except:
        sys.exit('Must be a number between 0 and 1 (exclusive)...')

min_size = 50
min_size_key = input(f'Enter min_size [{min_size}]: ')
if min_size_key != '':
    try:
        min_size = float(min_size_key)
        if min_size <= 0:
            sys.exit('Must be a number greater than 0...')
    except:
        sys.exit('Must be a number greater than 0...')
########################

outdir = 'outdir'
file_format = 'mol2'
if os.path.exists(outdir):
    os.rmdir(outdir)
os.mkdir(outdir)

ligand_coords_arr = predict(model, pdb, threshold=threshold, min_size=50)
for i, indices in enumerate(ligand_coords_arr):
    mol=openbabel.OBMol()
    for idx in indices:
        a=mol.NewAtom()
        a.SetVector(float(idx[0]),float(idx[1]),float(idx[2]))
    p_mol=pybel.Molecule(mol)
    p_mol.write(file_format, os.path.join(outdir, f'pocket{i}.{file_format}'), overwrite=True)
