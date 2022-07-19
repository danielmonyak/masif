import time
import os
import numpy as np
from IPython.core.debugger import set_trace
import sys
import importlib
from masif_modules.train_masif_site import run_masif_site, pad_indices
from default_config.masif_opts import masif_opts


# Apply mask to input_feat
def mask_input_feat(input_feat, mask):
    mymask = np.where(np.array(mask) == 0.0)[0]
    return np.delete(input_feat, mymask, axis=2)


#params = masif_opts["site"]
params = masif_opts['ligand_site']


#pdb = sys.argv[1]
pdb = '3O7W_A_'

# Shape precomputation dir.
parent_in_dir = params["masif_precomputation_dir"]
eval_list = []

# Build the neural network model
from masif_modules.MaSIF_site import MaSIF_site

learning_obj = MaSIF_site(
    params["max_distance"],
    n_thetas=4,
    n_rhos=3,
    n_rotations=4,
    idx_gpu="/gpu:0",
    feat_mask=params["feat_mask"],
    n_conv_layers=params["n_conv_layers"],
)
print("Restoring model from: " + params["model_dir"] + "model")
learning_obj.saver.restore(learning_obj.session, params["model_dir"] + "model")


if not os.path.exists(params["out_pred_dir"]):
    os.makedirs(params["out_pred_dir"])

in_dir = params["masif_precomputation_dir"] + pdb + "/"
pid = 'p1'
rho_wrt_center = np.load(in_dir + pid + "_rho_wrt_center.npy")
theta_wrt_center = np.load(in_dir + pid + "_theta_wrt_center.npy")
input_feat = np.load(in_dir + pid + "_input_feat.npy")
input_feat = mask_input_feat(input_feat, params["feat_mask"])
mask = np.load(in_dir + pid + "_mask.npy")
indices = np.load(in_dir + pid + "_list_indices.npy", encoding="latin1", allow_pickle=True)
indices = pad_indices(indices, mask.shape[1])

mask = np.expand_dims(mask, 2)
feed_dict = {
    learning_obj.rho_coords: rho_wrt_center,
    learning_obj.theta_coords: theta_wrt_center,
    learning_obj.input_feat: input_feat,
    learning_obj.mask: mask,
    learning_obj.indices_tensor: indices,
}

score = learning_obj.session.run(learning_obj.full_score, feed_dict=feed_dict)
outpath = os.path.join(params["out_pred_dir"], f'{pdb}y_pred.npy')
np.save(outpath, score)
#pocket_points_pred = (full_score > 0.5).nonzero()[0]
#outpath = os.path.join(params["out_pred_dir"], f'{pdb}pocket_points_pred.npy')
#np.save(outpath, pocket_points_pred)
