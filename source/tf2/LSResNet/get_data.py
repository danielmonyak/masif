import os
import numpy as np
from scipy import spatial
import default_config.util as util
from default_config.masif_opts import masif_opts
import tfbio.data

params = masif_opts["LSResNet"]

def get_data(func_input, training = True, make_y = True, mode='pdb_id', include_solvents=False):
    if include_solvents:
        ligand_list = masif_opts['all_ligands']
    else:
        ligand_list = masif_opts['ligand_list']
    
    if mode == 'pdb_id':
        pdb_id = func_input
        mydir = os.path.join(params["masif_precomputation_dir"], pdb_id.rstrip('_') + '_')
    elif mode == 'path':
        mydir = func_input
    else:
        sys.exit('Must provide a valid mode argument...')

    try:
        input_feat = np.load(os.path.join(mydir, "p1_input_feat.npy"))
        rho_coords = np.load(os.path.join(mydir, "p1_rho_wrt_center.npy"))
        theta_coords = np.load(os.path.join(mydir, "p1_theta_wrt_center.npy"))
        mask = np.load(os.path.join(mydir, "p1_mask.npy"))
        n_samples = mask.shape[0]

        if training and n_samples > 8000:
            return None
    except:
        return None
    
    mask = np.expand_dims(mask, 2)

    data_tsrs = tuple(np.expand_dims(tsr, axis=0) for tsr in [input_feat, rho_coords, theta_coords, mask])
    
    ###############################################################
    ###############################################################
    X_coords = np.load(os.path.join(mydir, "p1_X.npy"))
    Y_coords = np.load(os.path.join(mydir, "p1_Y.npy"))
    Z_coords = np.load(os.path.join(mydir, "p1_Z.npy"))
    xyz_coords = np.vstack([X_coords, Y_coords, Z_coords ]).T
    
    # Normalize coordinates
    centroid = xyz_coords.mean(axis=0)
    xyz_coords_normalized = xyz_coords - centroid
    
    if make_y:
        tree = spatial.KDTree(xyz_coords)
        
        coordsPath = os.path.join(
            params['ligand_coords_dir'], "{}_ligand_coords.npy".format(pdb_id.split("_")[0])
        )
        try:
            all_ligand_coords = np.load(coordsPath, allow_pickle=True, encoding='latin1')
            all_ligand_types = np.load(
                os.path.join(
                    params['ligand_coords_dir'], "{}_ligand_types.npy".format(pdb_id.split("_")[0])
                )
            ).astype(str)
        except:
            #print(f'Problem opening {coordsPath}')
            return None

        pocket_points = []
        for j, structure_ligand in enumerate(all_ligand_types):
            if not structure_ligand in ligand_list:
                continue

            ligand_coords = all_ligand_coords[j]
            temp_pocket_points = tree.query_ball_point(ligand_coords, 3.0)
            temp_pocket_points = list(set([pp for p in temp_pocket_points for pp in p]))
            pocket_points.extend(temp_pocket_points)

        labels = np.zeros([n_samples, 1], dtype=np.int32)
        labels[pocket_points, 0] = 1

        if (np.mean(labels) > 0.75) or (np.sum(labels) < 30):
            return None
    
        resolution = 1. / params['scale']
        y = tfbio.data.make_grid(xyz_coords_normalized, labels, max_dist=params['max_dist'], grid_resolution=resolution)
    
        y[y > 0] = 1
    else:
        y = None
        
    
    ###
    X = (data_tsrs, np.expand_dims(xyz_coords_normalized, axis=0))

    if training:
        return X, y
    
    return X, y, centroid
