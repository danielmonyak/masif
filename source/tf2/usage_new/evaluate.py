import os
import sys
import numpy as np
import pandas as pd
from scipy import spatial
from default_config.util import *
from tf2.usage.predictor import Predictor

params = masif_opts['ligand']
ligand_list = params['ligand_list']

ligand_coord_dir = params["ligand_coords_dir"]
precom_dir = params['masif_precomputation_dir']
binding_dir = '/data02/daniel/PUresNet/site_predictions'

pred = Predictor(ligand_model_path = '/home/daniel.monyak/software/masif/source/tf2/usage/masif_ligand_model/savedModel')

listDir = '/home/daniel.monyak/software/masif/data/masif_ligand/lists'
train_file = 'train_pdbs_sequence.npy'
val_file = 'val_pdbs_sequence.npy'
test_file = 'test_pdbs_sequence.npy'

train_list = np.char.add(np.load(os.path.join(listDir, train_file)).astype(str), '_')
val_list = np.char.add(np.load(os.path.join(listDir, val_file)).astype(str), '_')
test_list = np.char.add(np.load(os.path.join(listDir, test_file)).astype(str), '_')


pdb_list = []
pocket_list = []
dataset_list = []
recall_list = []
precision_list = []
ligandIdx_true_list = []
true_pts_ligandIdx_pred_list = []
pred_pts_ligandIdx_pred_list = []
npoints_true_list = []
npoints_pred_list = []

dataset_dict = {'train' : train_list, 'test' : test_list, 'val' : val_list}

for dataset in dataset_dict.keys():
    data = dataset_dict[dataset]
    n_data = len(data)
    for i, pdb in enumerate(data):
        print(f'{i} of {n_data} {dataset} pdbs running...')
        print(pdb)
        try:
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
        
        ####################
        n_ligands = len(all_ligand_types)
        
        pdb_dir = os.path.join(precom_dir, pdb)
        xyz_coords = Predictor.getXYZCoords(pdb_dir)
        tree = spatial.KDTree(xyz_coords)
        
        pp_true_list = 
        
        for lig_i in range(n_ligands):
            print(f'Pocket {lig_i}')
            
            ligand_coords = all_ligand_coords[lig_i]
            pocket_points_true = tree.query_ball_point(ligand_coords, 3.0)
            pocket_points_true = list(set([pp for p in pocket_points_true for pp in p]))

            npoints_true = len(pocket_points_true)

            if npoints_true == 0:
                print('Zero true pocket points...')
                continue

            ligand_true = all_ligand_types[lig_i]
            ligandIdx_true = ligand_list.index(ligand_true)

            ####################
            pred.loadData(pdb_dir)
            true_pts_ligandIdx_pred = pred.predictLigandIdx(pred.getLigandX(pocket_points_true)).numpy()

            ####################
            pdb_pnet_dir = os.path.join(binding_dir, pdb.rstrip("_"))
            files = os.listdir(pdb_pnet_dir)
            if len(files) == 0:
                print('Zero pockets were predicted...')
                continue

            n_pockets = np.sum(np.char.endswith(files, '.txt'))

            pocket_points_pred = []
            for pocket in range(n_pockets):
                pnet_coords = np.loadtxt(os.path.join(pdb_pnet_dir, f'pocket{pocket}.txt'), dtype=float)
                pp_pred_temp = tree.query_ball_point(pnet_coords, 3.0)
                pp_pred_temp = list(set([pp for p in pp_pred_temp for pp in p]))
                pocket_points_pred.extend(pp_pred_temp)

            npoints_pred = len(pocket_points_pred)

            pred_pts_ligandIdx_pred = pred.predictLigandIdx(pred.getLigandX(pocket_points_pred)).numpy()

            ####################
            overlap = np.intersect1d(pocket_points_true, pocket_points_pred)
            recall = len(overlap)/npoints_true
            try:
                precision = len(overlap)/npoints_pred
            except:
                precision = np.NaN

            ###############
            print(f'Recall: {recall}')
            print(f'Precision: {precision}')
            print(f'True ligand: {ligandIdx_true}')
            print(f'Prediction from true pocket points: {true_pts_ligandIdx_pred}')
            print(f'Prediction from predicted pocket points: {pred_pts_ligandIdx_pred}')

            pdb_list.append(pdb)
            pocket_list.append(lig_i)
            dataset_list.append(dataset)
            recall_list.append(recall)
            precision_list.append(precision)
            ligandIdx_true_list.append(ligandIdx_true)
            true_pts_ligandIdx_pred_list.append(true_pts_ligandIdx_pred)
            pred_pts_ligandIdx_pred_list.append(pred_pts_ligandIdx_pred)
            npoints_true_list.append(npoints_true)
            npoints_pred_list.append(npoints_pred)

columns = ['pdb_list', 'pocket_list', 'dataset_list', 'recall_list', 'precision_list', 'ligandIdx_true_list', 'true_pts_ligandIdx_pred_list', 'pred_pts_ligandIdx_pred_list', 'npoints_true_list', 'npoints_pred_list']
results = pd.DataFrame(dict(zip([col.partition('_list')[0] for col in columns], [eval(col) for col in columns])))
results.to_csv('results.csv', index=False)

