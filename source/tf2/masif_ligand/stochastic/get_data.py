import os
import numpy as np
from scipy import spatial
from default_config.util import *

params = masif_opts["ligand"]
minPockets = params['minPockets']
ligand_list = masif_opts['all_ligands']

def get_data(pdb_id):
    mydir = os.path.join(params["masif_precomputation_dir"], pdb_id.rstrip('_') + '_')
    
    try:
        input_feat = np.load(os.path.join(mydir, "p1_input_feat.npy"))
        rho_coords = np.load(os.path.join(mydir, "p1_rho_wrt_center.npy"))
        theta_coords = np.load(os.path.join(mydir, "p1_theta_wrt_center.npy"))
        mask = np.load(os.path.join(mydir, "p1_mask.npy"))
    except:
        return None
    
    mask = np.expand_dims(mask, 2)
    n_samples = mask.shape[0]
    
    X = tuple(np.expand_dims(tsr, axis=0) for tsr in [input_feat, rho_coords, theta_coords, mask])
    
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
        all_ligand_coords = np.load(coordsPath, allow_pickle=True, encoding='latin1')
    except:
        #print(f'Problem opening {coordsPath}')
        return None

    all_ligand_types = np.load(
        os.path.join(
            params['ligand_coords_dir'], "{}_ligand_types.npy".format(pdb_id.split("_")[0])
        )
    ).astype(str)
    
    pocket_points = []
    y = []
    for j, structure_ligand in enumerate(all_ligand_types):
        ligIdx = ligand_list.index(structure_ligand)
        if ligIdx < 7:
            dist = 3.0
        else:
            dist = 7.0
        ligand_coords = all_ligand_coords[j]
        temp_pocket_points = tree.query_ball_point(ligand_coords, dist)
        temp_pocket_points = list(set([pp for p in temp_pocket_points for pp in p]))
        temp_npoints = len(temp_pocket_points)
        if (temp_npoints > minPockets) and (temp_npoints/n_samples < 0.75):
            pocket_points.append(temp_pocket_points)
            y.append(ligIdx)

    if len(pocket_points) == 0:
        #print(f'{pdb_id} has no pockets big enough...')
        return None
    
    return X, pocket_points, y
