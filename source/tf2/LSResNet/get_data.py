import os
import numpy as np
from scipy import spatial
from default_config.util import *
import tfbio.data

params = masif_opts["ligand_site"]

def get_data(pdb_id):
    mydir = os.path.join(params["masif_precomputation_dir"], pdb_id + '_')

    mask = np.load(os.path.join(mydir, "p1_mask.npy"))
    n_samples = mask.shape[0]

    if n_samples > 8000:
        return None

    input_feat = np.load(os.path.join(mydir, "p1_input_feat.npy"))
    rho_coords = np.load(os.path.join(mydir, "p1_rho_wrt_center.npy"))
    theta_coords = np.load(os.path.join(mydir, "p1_theta_wrt_center.npy"))
    mask = np.expand_dims(mask, 2)

    data_tsrs = tuple(np.expand_dims(tsr, axis=0) for tsr in [input_feat, rho_coords, theta_coords, mask])
    
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
        return None

    pocket_points = []
    for j, structure_ligand in enumerate(all_ligand_coords):
        ligand_coords = all_ligand_coords[j]
        temp_pocket_points = tree.query_ball_point(ligand_coords, 3.0)
        temp_pocket_points = list(set([pp for p in temp_pocket_points for pp in p]))
        pocket_points.extend(temp_pocket_points)

    labels = np.zeros([n_samples, 1], dtype=np.int32)
    labels[pocket_points, 0] = 1

    if (np.mean(labels) > 0.75) or (np.sum(labels) < 30):
        return None
    
    scale = 0.5
    resolution = 1. / scale
    y = tfbio.data.make_grid(xyz_coords, labels, max_dist=model.max_dist, grid_resolution=resolution)
    
    y[y > 0] = 1
    
    X = (data_tsrs, np.expand_dims(xyz_coords, axis=0))

    return X, y
