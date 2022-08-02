import os
import numpy as np
from scipy import spatial
from default_config.util import *

params = masif_opts["ligand"]
ligand_list = masif_opts['all_ligands']
pos_dists = [3.0, 5.0, 7.0, 9.0]

freq_dict = dict(zip(ligand_list, [dict(zip(pos_dists, [[] for j in range(len(pos_dists))])) for i in range(len(ligand_list))]))

pdb_list = os.listdir(params["masif_precomputation_dir"])

bad_coords = []
wrong_ligands = []
no_precomp = []

n_pdbs = len(pdb_list)
for k, pdb_id in enumerate(pdb_list):
    print(f'Working on {k} of {n_pdbs} proteins...')
    mydir = os.path.join(params["masif_precomputation_dir"], pdb_id.rstrip('_') + '_')
    try:
        X_coords = np.load(os.path.join(mydir, "p1_X.npy"))
        Y_coords = np.load(os.path.join(mydir, "p1_Y.npy"))
        Z_coords = np.load(os.path.join(mydir, "p1_Z.npy"))
    except:
        no_precomp.append(pdb_id)
        continue

    xyz_coords = np.vstack([X_coords, Y_coords, Z_coords ]).T
    tree = spatial.KDTree(xyz_coords)
    coordsPath = os.path.join(
        params['ligand_coords_dir'], "{}_ligand_coords.npy".format(pdb_id.split("_")[0])
    )
    try:
        all_ligand_coords = np.load(coordsPath, allow_pickle=True, encoding='latin1')
    except:
        bad_coords.append(pdb_id)
        continue

    all_ligand_types = np.load(
        os.path.join(
            params['ligand_coords_dir'], "{}_ligand_types.npy".format(pdb_id.split("_")[0])
        )
    ).astype(str)

    for j, structure_ligand in enumerate(all_ligand_types):
        if not structure_ligand in ligand_list:
            wrong_ligands.append(pdb_id)
            continue

        ligIdx = ligand_list.index(structure_ligand)
        
        ligand_coords = all_ligand_coords[j]
        for dist in pos_dists:
            temp_pocket_points = tree.query_ball_point(ligand_coords, dist)
            temp_pocket_points = list(set([pp for p in temp_pocket_points for pp in p]))
            temp_npoints = len(temp_pocket_points)
            freq_dict[structure_ligand][dist].append(temp_npoints)

