import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import sys
import numpy as np
from scipy import spatial
from random import shuffle
from default_config.util import *
import tensorflow as tf

params = masif_opts["ligand"]

####
#ligand_list = masif_opts['ligand_list']
ligand_list = masif_opts['all_ligands']
n_classes = len(ligand_list)
####

angstrom_dist = 7.0


all_pdbs = os.listdir(params["masif_precomputation_dir"])

# Structures are randomly assigned to train, validation and test sets
shuffle(all_pdbs)
train = int(len(all_pdbs) * params["train_fract"])
val = int(len(all_pdbs) * params["val_fract"])
test = int(len(all_pdbs) * params["test_fract"])
print("Train", train)
print("Validation", val)
print("Test", test)
train_pdbs = all_pdbs[:train]
val_pdbs = all_pdbs[train : train + val]
test_pdbs = all_pdbs[train + val : train + val + test]


# Edited by Daniel Monyak
train_pdbs = np.char.rstrip(train_pdbs, '_')
val_pdbs = np.char.rstrip(val_pdbs, '_')
test_pdbs = np.char.rstrip(test_pdbs, '_')
####

####
listdir = 'extraLigands_lists'
if not os.path.exists(listdir):
    os.mkdir(listdir)
####


# Uncommented these save statements so that the lists would be redone
np.save(os.path.join(listdir, 'train_pdbs_sequence.npy'), train_pdbs)
np.save(os.path.join(listdir, 'val_pdbs_sequence.npy'), val_pdbs)
np.save(os.path.join(listdir, 'test_pdbs_sequence.npy'), test_pdbs)

# For this run use the train, validation and test sets actually used
train_pdbs = np.load(os.path.join(listdir, 'train_pdbs_sequence.npy')).astype(str)
val_pdbs = np.load(os.path.join(listdir, 'val_pdbs_sequence.npy')).astype(str)
test_pdbs = np.load(os.path.join(listdir, 'test_pdbs_sequence.npy')).astype(str)

precom_dir = params["masif_precomputation_dir"]
ligand_coord_dir = params["ligand_coords_dir"]

defaultCode = params['defaultCode']
minPockets = params['minPockets']

outdir = '/data02/daniel/masif/datasets/tf2/masif_ligand/extraLigands'
if not os.path.exists(outdir):
    os.mkdir(outdir)

genOutPath = os.path.join(outdir, '{}_{}.npy')

def helper(feed_dict):
    flat_list = list(map(lambda tsr_key : feed_dict[tsr_key].flatten(), data_order))
    return np.concatenate(flat_list, axis = 0)

def compile_and_save(feed_list, y_list, dataset, len_list):
    tsr_list = list(map(helper, feed_list))
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
    i = 0
    
    feed_list = []
    y_list = []
    len_list = []

    temp_data = dataset_list[dataset]
    n_pdbs = len(temp_data)
    for i, pdb in enumerate(temp_data):
        print(f'{dataset} record {i+1} of {n_pdbs}, {pdb}')
        try:
            # Load precomputed data
            input_feat = np.load(
                os.path.join(precom_dir, pdb + "_", "p1_input_feat.npy")
            )
            rho_wrt_center = np.load(
                os.path.join(precom_dir, pdb + "_", "p1_rho_wrt_center.npy")
            )
            theta_wrt_center = np.load(
                os.path.join(precom_dir, pdb + "_", "p1_theta_wrt_center.npy")
            )
            mask = np.load(
                os.path.join(precom_dir, pdb + "_", "p1_mask.npy")
            )
            X = np.load(os.path.join(precom_dir, pdb + "_", "p1_X.npy"))
            Y = np.load(os.path.join(precom_dir, pdb + "_", "p1_Y.npy"))
            Z = np.load(os.path.join(precom_dir, pdb + "_", "p1_Z.npy"))
            all_ligand_coords = np.load(
                os.path.join(
                    ligand_coord_dir, "{}_ligand_coords.npy".format(pdb.split("_")[0])
                )
            )
            all_ligand_types = np.load(
                os.path.join(
                    ligand_coord_dir, "{}_ligand_types.npy".format(pdb.split("_")[0])
                )
            ).astype(str)
        except:
            continue
        
        if len(all_ligand_types) == 0:
            continue  
        
        xyz_coords = np.vstack([X, Y, Z]).T
        tree = spatial.KDTree(xyz_coords)
        pocket_labels = np.zeros(
            (xyz_coords.shape[0], len(all_ligand_types)), dtype=np.int32
        )
        # Label points on surface within 3A distance from ligand with corresponding ligand type
        for j, structure_ligand in enumerate(all_ligand_types):
            ligand_coords = all_ligand_coords[j]
            pocket_points = tree.query_ball_point(ligand_coords, angstrom_dist)
            pocket_points = list(set([pp for p in pocket_points for pp in p]))
            npoints = len(pocket_points)
            
            if npoints < minPockets:
                continue
            #
            label = ligand_list.index(structure_ligand)
            print(f'Ligand: {label}')
            #
            pocket_labels = np.zeros(n_classes, dtype=np.float32)
            pocket_labels[label] = 1.0
            #
            feed_dict = {
                'input_feat' : input_feat[pocket_points],
                'rho_coords' : rho_wrt_center[pocket_points],
                'theta_coords' : theta_wrt_center[pocket_points],
                'mask' : mask[pocket_points]
            }
            feed_list.append(feed_dict)
            y_list.append(pocket_labels)
            len_list.append(feed_dict['input_feat'].size + feed_dict['rho_coords'].size + feed_dict['theta_coords'].size + feed_dict['mask'].size)
        
    compile_and_save(feed_list, y_list, dataset, len_list)

print('Finished!')
