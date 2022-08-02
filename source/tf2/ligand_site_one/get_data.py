import os
import numpy as np
from scipy import spatial
from default_config.util import *

params = masif_opts["ligand_site"]

def get_data(pdb_id, training = True, make_y=True):
    mydir = os.path.join(params["masif_precomputation_dir"], pdb_id.rstrip('_') + '_')

    mask = np.load(os.path.join(mydir, "p1_mask.npy"))
    n_samples = mask.shape[0]

    if training and n_samples > 8000:
        return None

    input_feat = np.load(os.path.join(mydir, "p1_input_feat.npy"))
    rho_coords = np.load(os.path.join(mydir, "p1_rho_wrt_center.npy"))
    theta_coords = np.load(os.path.join(mydir, "p1_theta_wrt_center.npy"))
    mask = np.expand_dims(mask, 2)
    indices = np.load(os.path.join(mydir, "p1_list_indices.npy"), encoding="latin1", allow_pickle = True)
    # indices is (n_verts x <30), it should be
    indices = pad_indices(indices, mask.shape[1]).astype(np.int32)

    data_tsrs = tuple(np.expand_dims(tsr, axis=0) for tsr in [input_feat, rho_coords, theta_coords, mask])
    indices = np.expand_dims(indices, axis=0)
    X = (data_tsrs, indices)

    ###############################################################
    ###############################################################
    if make_y:
        X_coords = np.load(os.path.join(mydir, "p1_X.npy"))
        Y_coords = np.load(os.path.join(mydir, "p1_Y.npy"))
        Z_coords = np.load(os.path.join(mydir, "p1_Z.npy"))
        xyz_coords = np.vstack([X_coords, Y_coords, Z_coords ]).T
        tree = spatial.KDTree(xyz_coords)
        coordsPath = os.path.join(
            params['ligand_coords_dir'], "{}_ligand_coords.npy".format(pdb_id.split("_")[0])
        )
        try:
            all_ligand_coords = np.load(coordsPath, allow_pickle=True, encoding='latin1')
        except:
            print(f'Problem opening {coordsPath}')
            return None

        pocket_points = []
        for j, structure_ligand in enumerate(all_ligand_coords):
            ligand_coords = all_ligand_coords[j]
            temp_pocket_points = tree.query_ball_point(ligand_coords, 3.0)
            temp_pocket_points = list(set([pp for p in temp_pocket_points for pp in p]))
            if len(temp_pocket_points) > 32:
                pocket_points.extend(temp_pocket_points)

        if len(pocket_points) == 0:
            print(f'{pdb_id} has no pockets big enough...')
            return None

        y = np.zeros([1, n_samples, 1], dtype=np.int32)
        y[0, pocket_points, 0] = 1

        if np.mean(y) > 0.75:
            print(f'{pdb_id} is weird...')
            return None

        n_pockets = np.sum(y)
        n_empty = n_samples - n_pockets

        sample_weight = np.empty(shape=y.shape, dtype=np.float32)
        sample_weight.fill(n_samples/(2*n_empty))
        sample_weight[0, pocket_points, 0] = n_samples/(2*n_pockets)
    else:
        y = sample_weight = None

    return X, y, sample_weight
