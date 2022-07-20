import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import sys
import importlib
from IPython.core.debugger import set_trace
import pickle
import numpy as np
from scipy import spatial
import tensorflow as tf

phys_gpus = tf.config.list_physical_devices('GPU')
for phys_g in phys_gpus:
    tf.config.experimental.set_memory_growth(phys_g, True)

from default_config.util import *
from tf2.ligand_site_one.sep_layers.MaSIF_ligand_site_one import MaSIF_ligand_site

#############################################
continue_training = False
#read_metrics = False

starting_sample = 0
starting_epoch = 0
#############################################

#params = masif_opts["ligand"]
params = masif_opts["ligand_site"]

modelDir = 'kerasModel'
ckpPath = os.path.join(modelDir, 'ckp')
ckpStatePath = ckpPath + '.pickle'

modelPath_endTraining = os.path.join(modelDir, 'savedModel_endTraining')

#############################################
#############################################
num_epochs = 20                 #############
pdb_ckp_thresh = 10             #############
#############################################
#############################################

model = MaSIF_ligand_site(
    params["max_distance"],
    feat_mask=params["feat_mask"],
    n_thetas=4,
    n_rhos=3,
    learning_rate = 1e-3,
    n_rotations=4,
    reg_val = 1e-4,
    reg_type = 'l2'
)

from_logits = model.loss_fn.get_config()['from_logits']
binAcc = tf.keras.metrics.BinaryAccuracy(threshold = (not from_logits) * 0.5)
auc = tf.keras.metrics.AUC(from_logits = from_logits)

model.compile(optimizer = model.opt,
  loss = model.loss_fn,
  metrics=[binAcc, auc]
)

if continue_training:
    model.load_weights(ckpPath)
    print(f'Loaded model from {ckpPath}')

'''
if read_metrics:
    with open(ckpStatePath, 'rb') as handle:
        ckpState = pickle.load(handle)
    starting_epoch = ckpState['last_epoch']
    print(f'Resuming epoch {i} of training\nValidation accuracy: {best_acc}')
else:
    i = 0
    best_acc = 0'''

training_list = np.load('/home/daniel.monyak/software/masif/data/masif_ligand/lists/train_pdbs_sequence.npy')

#######################################
#######################################
#######################################

i = starting_epoch

print(f'Running training data, epoch {i}')
while i < num_epochs:
    train_j = 0
    #############################################################
    ###################     TRAINING DATA     ###################
    #############################################################
    for pdb_id in training_list:
        print(f'Epoch {i}, train pdb {train_j}, {pdb_id}')
        
        mydir = os.path.join(params["masif_precomputation_dir"], pdb_id + '_')
        
        mask = np.load(os.path.join(mydir, "p1_mask.npy"))
        n_samples = mask.shape[0]

        if n_samples > 8000:
            print('Too many patches to train on...')
            continue

        input_feat = np.load(os.path.join(mydir, "p1_input_feat.npy"))
        theta_wrt_center = np.load(os.path.join(mydir, "p1_theta_wrt_center.npy"))
        rho_wrt_center = np.load(os.path.join(mydir, "p1_rho_wrt_center.npy"))
        mask = np.expand_dims(mask, 2)
        indices = np.load(os.path.join(mydir, "p1_list_indices.npy"), encoding="latin1", allow_pickle = True)
        # indices is (n_verts x <30), it should be
        indices = pad_indices(indices, mask.shape[1]).astype(np.int32)

        data_tsrs = tuple(np.expand_dims(tsr, axis=0) for tsr in [input_feat, rho_wrt_center, theta_wrt_center, mask])
        indices = np.expand_dims(indices, axis=0)

        X = (data_tsrs, indices)

        ###############################################################
        ###############################################################
        X_coords = np.load(os.path.join(mydir, "p1_X.npy"))
        Y_coords = np.load(os.path.join(mydir, "p1_Y.npy"))
        Z_coords = np.load(os.path.join(mydir, "p1_Z.npy"))
        xyz_coords = np.vstack([X_coords, Y_coords, Z_coords ]).T
        tree = spatial.KDTree(xyz_coords)
        coordsPath = os.path.join(
            params['ligand_coords_dir'], "{}_ligand_coords.npy".format(pdb_id.split("_")[0])
        )
        try:
            all_ligand_coords = np.load(coordsPath)
        except:
            print(f'Problem opening {coordsPath}')
            continue
        pocket_points = []
        for j, structure_ligand in enumerate(all_ligand_coords):
            ligand_coords = all_ligand_coords[j]
            temp_pocket_points = tree.query_ball_point(ligand_coords, 3.0)
            temp_pocket_points = list(set([pp for p in temp_pocket_points for pp in p]))
            pocket_points.extend(temp_pocket_points)

        y = np.zeros([1, n_samples, 1], dtype=np.int32)
        y[0, pocket_points, 0] = 1

        if (np.mean(y) > 0.75) or (np.sum(y) < 30):
            print('Too many pocket points or not enough patches...')
            continue

        # TRAIN MODEL
        ################################################
        model.fit(X, y, verbose = 2)    ################
        ################################################
        
        train_j += 1

        if train_j % pdb_ckp_thresh == 0:
            print(f'Saving model weights to {ckpPath}')
            model.save_weights(ckpPath)

    train_j = 0
    i += 1

    print(f'Saving model weights to {ckpPath}')
    model.save_weights(ckpPath)

print(f'Saving model to to {modelPath_endTraining}')
model.save(modelPath_endTraining)
