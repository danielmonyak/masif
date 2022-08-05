import os
import numpy as np
from scipy import spatial
import default_config.util as util
from default_config.masif_opts import masif_opts
import tfbio.data

params = masif_opts['LSResNet']
ligand_list = masif_opts['ligand_list']

include_solvents = False

if include_solvents:
    ligand_list = masif_opts['all_ligands']
    all_pdbs = np.loadtxt('/home/daniel.monyak/software/masif/data/masif_ligand/newPDBs/filtered_pdbs.txt', dtype=str)
else:
    ligand_list = masif_opts['ligand_list']
    all_pdbs = np.loadtxt('/home/daniel.monyak/software/masif/data/masif_ligand/newPDBs/using_pdbs_final_reg.txt', dtype=str)

n_pdbs = len(all_pdbs)
for i, pdb_id in enumerate(all_pdbs):
    print(f'{i+1} of {n_pdbs} PDBs, {pdb_id}')
    
    mydir = os.path.join(params["masif_precomputation_dir"], pdb_id.rstrip('_') + '_')
    try:
        X_coords = np.load(os.path.join(mydir, "p1_X.npy"))
        Y_coords = np.load(os.path.join(mydir, "p1_Y.npy"))
        Z_coords = np.load(os.path.join(mydir, "p1_Z.npy"))
        xyz_coords = np.vstack([X_coords, Y_coords, Z_coords ]).T

        coordsPath = os.path.join(
            params['ligand_coords_dir'], "{}_ligand_coords.npy".format(pdb_id.split("_")[0])
        )
        all_ligand_coords = np.load(coordsPath, allow_pickle=True, encoding='latin1')
        all_ligand_types = np.load(
            os.path.join(
                params['ligand_coords_dir'], "{}_ligand_types.npy".format(pdb_id.split("_")[0])
            )
        ).astype(str)
    except:
        continue
    
    tree = spatial.KDTree(xyz_coords)
    n_samples = len(X_coords)
    
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
        continue
    
    resolution = 1. / params['scale']
    y = tfbio.data.make_grid(xyz_coords, labels, max_dist=params['max_dist'], grid_resolution=resolution)
    y[y > 0] = 1

    np.save(os.path.join(mydir, f'LSRN_y.npy'), y)

print('Finished!')
