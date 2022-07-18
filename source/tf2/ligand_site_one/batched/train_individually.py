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
from tf2.read_ligand_tfrecords import _parse_function
from MaSIF_ligand_site_one import MaSIF_ligand_site

dev = '/GPU:1'
cpu = '/CPU:0'
'''
log_gpus = tf.config.list_logical_devices('GPU')
gpu_strs = [g.name for g in log_gpus]
strategy = tf.distribute.MirroredStrategy(gpu_strs)
'''

#tf.config.set_soft_device_placement(True)
#tf.debugging.set_log_device_placement(True)

#############################################
continue_training = False
#read_metrics = False

starting_sample = 0
starting_epoch = 0
#############################################

#params = masif_opts["ligand"]
params = masif_opts["site"]

modelDir = 'kerasModel'
ckpPath = os.path.join(modelDir, 'ckp')
ckpStatePath = ckpPath + '.pickle'

modelPath_endTraining = os.path.join(modelDir, 'savedModel_endTraining')

#############################################
#############################################
num_epochs = 5                  #############
pdb_ckp_thresh = 10             #############
#############################################
#############################################

cv_batch_sz = 100

#with tf.device(dev):
#with strategy.scope():
model = MaSIF_ligand_site(
    params["max_distance"],
    feat_mask=params["feat_mask"],
    n_conv_layers = params['n_conv_layers'],
    conv_batch_size = cv_batch_sz
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
    best_acc = 0
'''

training_list = np.char.rstrip(np.loadtxt(params["training_list"]))
testing_list = np.char.rstrip(np.loadtxt(params["testing_list"]))

data_dirs = os.listdir(params["masif_precomputation_dir"])
np.random.shuffle(data_dirs)
data_dirs = data_dirs
n_val = len(data_dirs) // 10
val_dirs = set(data_dirs[(len(data_dirs) - n_val) :])


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
    for pdb_dir in data_dirs:
        print(f'Epoch {i}, train record {train_j}')
        
        mydir = os.path.join(params["masif_precomputation_dir"], pdb_idr)
        
        rho_wrt_center = np.load(os.path.join(mydir, "p1_rho_wrt_center.npy"))

        n_samples = rho_wrt_center.shape[0]
        # Memory limitation?
        if n_samples > 8000:
            train_j += 1
            continue

        theta_wrt_center = np.load(os.path.join(mydir, "p1_theta_wrt_center.npy"))
        input_feat = np.load(os.path.join(mydir, "p1_input_feat.npy"))
        mask = np.load(os.path.join(mydir, "p1_mask.npy"))
        mask = np.expand_dims(mask, 2)
        indices = np.load(os.path.join(mydir, "p1_list_indices.npy", encoding="latin1", allow_pickle = True))
        # indices is (n_verts x <30), it should be
        indices = pad_indices(indices, mask.shape[1]).astype(np.int32)

        data_tsrs = tuple(np.expand_dims(tsr, axis=0) for tsr in [input_feat, rho_wrt_center, theta_wrt_center, mask])
        indices = np.expand_dims(indices, axis=0)

        X = (data_tsrs, indices)

        ###############################################################
        ###############################################################
        X_coords = np.load(os.path.join(precom_dir, pdb + "_", "pid" + "_X.npy"))
        Y_coords = np.load(os.path.join(precom_dir, pdb + "_", "pid" + "_Y.npy"))
        Z_coords = np.load(os.path.join(precom_dir, pdb + "_", "pid" + "_Z.npy"))
        xyz_coords = np.vstack([X, Y, Z]).T
        tree = spatial.KDTree(xyz_coords)
        all_ligand_coords = np.load(
            os.path.join(
                ligand_coord_dir, "{}_ligand_coords.npy".format(pdb.split("_")[0])
            )
        )
        pocket_points = []
        for j, structure_ligand in enumerate(all_ligand_coords):
            ligand_coords = all_ligand_coords[j]
            temp_pocket_points = tree.query_ball_point(ligand_coords, 3.0)
            temp_pocket_points = list(set([pp for p in temp_pocket_points for pp in p]))
            pocket_points.extend(temp_pocket_points)

        y = np.zeros([1, n_samples], dtype=np.int32)
        y[0, pocket_points] = 1

        if (np.mean(y) > 0.75 or np.sum(y) < 30):
            train_j += 1
            continue

        # TRAIN MODEL
        ################################################
        model.fit(X, y, verbose = 2)    ################
        ################################################
        
        print('\n\nFinished training on one protein\n\n')
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
