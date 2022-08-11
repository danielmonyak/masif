import os
import numpy as np
from scipy import spatial
from default_config.util import *

params = masif_opts["ligand"]
minPockets = params['minPockets']
#

#from time import time

def get_data(pdb_id, include_solvents=False):
    if include_solvents:
        ligand_list = masif_opts['all_ligands']
    else:
        ligand_list = masif_opts['ligand_list']
    
    mydir = os.path.join(params["masif_precomputation_dir"], pdb_id.rstrip('_') + '_')
    
    try:
#        before = time()
        
        input_feat = np.load(os.path.join(mydir, "p1_input_feat.npy"))
        rho_coords = np.load(os.path.join(mydir, "p1_rho_wrt_center.npy"))
        theta_coords = np.load(os.path.join(mydir, "p1_theta_wrt_center.npy"))
        mask = np.load(os.path.join(mydir, "p1_mask.npy"))
        
        coordsPath = os.path.join(
            params['ligand_coords_dir'], "{}_ligand_coords.npy".format(pdb_id.split("_")[0])
        )
        all_ligand_coords = np.load(coordsPath, allow_pickle=True, encoding='latin1')
        all_ligand_types = np.load(
            os.path.join(
                params['ligand_coords_dir'], "{}_ligand_types.npy".format(pdb_id.split("_")[0])
            )
        ).astype(str)
#        print('import data %.4f' % (time() - before))
    except:
        return None
    
    mask = np.expand_dims(mask, 2)
    n_samples = mask.shape[0]
    
    X = tuple(np.expand_dims(tsr, axis=0) for tsr in [input_feat, rho_coords, theta_coords, mask])
    
    ###############################################################
    ###############################################################
#    import_2_time = -time()
    
    X_coords = np.load(os.path.join(mydir, "p1_X.npy"))
    Y_coords = np.load(os.path.join(mydir, "p1_Y.npy"))
    Z_coords = np.load(os.path.join(mydir, "p1_Z.npy"))
    xyz_coords = np.vstack([X_coords, Y_coords, Z_coords ]).T
    tree = spatial.KDTree(xyz_coords)
    '''coordsPath = os.path.join(
        params['ligand_coords_dir'], "{}_ligand_coords.npy".format(pdb_id.split("_")[0])
    )
    try:
        all_ligand_coords = np.load(coordsPath, allow_pickle=True, encoding='latin1')
    except:
        #print(f'Problem opening {coordsPath}')
        return None
    '''
    
#    import_2_time += time()
#    print('import data 2 %.4f' % import_2_time)
    
#    prep_pp_time = -time()
    
    pocket_points = []
    y = []
    for j, structure_ligand in enumerate(all_ligand_types):
        if not structure_ligand in ligand_list:
            continue

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
    
    
#    prep_pp_time += time()
#    print('prep_pp_time %.4f' % prep_pp_time)
    
    if len(pocket_points) == 0:
        #print(f'{pdb_id} has no pockets big enough...')
        return None
    
    return X, pocket_points, y
