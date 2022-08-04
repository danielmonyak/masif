import os
import numpy as np
import default_config.util as util
from default_config.masif_opts import masif_opts
from tf2.masif_ligand.stochastic.get_data import get_data

params = masif_opts["ligand"]

include_solvents = False

train_pdbs = np.load('/home/daniel.monyak/software/masif/data/masif_ligand/newPDBs/lists/train_reg.npy')
val_pdbs = np.load('/home/daniel.monyak/software/masif/data/masif_ligand/newPDBs/lists/val_reg.npy')
test_pdbs = np.load('/home/daniel.monyak/software/masif/data/masif_ligand/newPDBs/lists/test_reg.npy')

precom_dir = params["masif_precomputation_dir"]
ligand_coord_dir = params["ligand_coords_dir"]

defaultCode = params['defaultCode']
minPockets = params['minPockets']

outdir = '/data02/daniel/masif/datasets/tf2/masif_ligand/allReg'
if not os.path.exists(outdir):
    os.mkdir(outdir)

genOutPath = os.path.join(outdir, '{}_{}.npy')

def helper(X_temp):
    flat_list = [arr.flatten() for arr in X_temp]
    return np.concatenate(flat_list, axis = 0)

def compile_and_save(X_list, y_list, dataset, len_list):
    tsr_list = list(map(helper, X_list))
    #
    X = np.empty([len(tsr_list), max(len_list)])
    X.fill(defaultCode)
    for i, tsr in enumerate(tsr_list):
        X[i, :len(tsr)] = tsr
    #
    y = np.stack(y_list)
    np.save(genOutPath.format(dataset, 'X'), X)
    np.save(genOutPath.format(dataset, 'y'), y)


dataset_list = {'train' : train_pdbs, 'val' : val_pdbs, 'test' : test_pdbs}

for dataset in dataset_list.keys():
    X_list = []
    y_list = []
    len_list = []

    temp_data = dataset_list[dataset]
    n_pdbs = len(temp_data)
    for i, pdb_id in enumerate(temp_data):
        print(f'{dataset} record {i+1} of {n_pdbs}, {pdb_id}')
        data = get_data(pdb_id, include_solvents)
        if data is None:
            continue
        
        X, pocket_points, y = data
        for k, pp in enumerate(pocket_points):
            y_temp = y[k]
            X_temp = [arr[:, pp] for arr in X]
            
            X_list.append(X_temp)
            y_list.append(y_temp)
            len_list.append(sum([arr.size for arr in X_temp]))

    compile_and_save(X_list, y_list, dataset, len_list)

print('Finished!')
