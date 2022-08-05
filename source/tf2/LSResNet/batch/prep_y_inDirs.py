import os
import numpy as np
import default_config.util as util
from default_config.masif_opts import masif_opts
from tf2.masif_ligand.stochastic.get_data import get_data

params = masif_opts["ligand"]
precom_dir = params["masif_precomputation_dir"]

include_solvents = True

all_pdbs = np.loadtxt('/home/daniel.monyak/software/masif/data/masif_ligand/newPDBs/filtered_pdbs.txt', dtype=str)
n_pdbs = len(all_pdbs)
for i, pdb_id in enumerate(all_pdbs):
    print(f'{i+1} of {n_pdbs} PDBs, {pdb_id}')
    data = get_data(pdb_id, include_solvents)
    if data is None:
        continue

    mydir = os.path.join(precom_dir, pdb_id)
    X, pocket_points, y = data
    
    ###
    X_list = []
    y_list = []
    ###
    for k, pp in enumerate(pocket_points):
        y_temp = y[k]
        X_temp = [arr[:, pp] for arr in X]
        
        X_temp[0] = np.squeeze(X_temp[0], 0)
        X_temp[3] = np.squeeze(X_temp[3], 0)
        
        ###
        X_list.append(X_temp)
        y_list.append(y_temp)
        ###
        
        #np.save(os.path.join(mydir, f'X_{k}.npy'), X_temp)
        #np.save(os.path.join(mydir, f'y_{k}.npy'), y_temp)
    
    np.save(os.path.join(mydir, f'X.npy'), X_list)
    np.save(os.path.join(mydir, f'y.npy'), y_list)

print('Finished!')
