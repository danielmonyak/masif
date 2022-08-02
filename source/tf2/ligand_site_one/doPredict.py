import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import numpy as np
import tensorflow as tf

phys_gpus = tf.config.list_physical_devices('GPU')
for phys_g in phys_gpus:
    tf.config.experimental.set_memory_growth(phys_g, True)

import default_config.util as util
from default_config.masif_opts import masif_opts
from tf2.ligand_site_one.predict import predict
from tf2.ligand_site_one.MaSIF_ligand_site_one import MaSIF_ligand_site


########################
pdb = input(f'Enter pdb: ')
if pdb == '':
    sys.exit('Must enter a valid pdb...')

modelDir = 'kerasModel'
modelDir_key = input(f'Enter directory with model checkpoint [{modelDir}]: ')
if modelDir_key != '':
    modelDir = modelDir_key
ckpPath = os.path.join(modelDir, 'ckp')

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

make_y_key = input(f'Collect true values? ([y]/n): ')
if (make_y_key == '') or (make_y_key == 'y'):
    make_y = True
elif make_y_key == 'n':
    make_y = False
else:
    sys.exit('Please enter a valid response...')
########################


model = MaSIF_ligand_site(
    params["max_distance"],
    feat_mask=params["feat_mask"],
    n_thetas=4,
    n_rhos=3,
    n_rotations=4
)
load_status = model.load_weights(ckpPath)
load_status.expect_partial()


outdir = 'outdir'
file_format = 'mol2'
if os.path.exists(outdir):
    _ = os.system(f'rm -r {outdir}')
os.mkdir(outdir)

ligand_coords_arr = predict(model, pdb, threshold, min_size, make_y)
for i, indices in enumerate(ligand_coords_arr):
    mol=openbabel.OBMol()
    for idx in indices:
        a=mol.NewAtom()
        a.SetVector(float(idx[0]),float(idx[1]),float(idx[2]))
    p_mol=pybel.Molecule(mol)
    p_mol.write(file_format, os.path.join(outdir, f'pocket{i}.{file_format}'))
    np.save(os.path.join(outdir, f'pocket{i}.npy'), indices)
