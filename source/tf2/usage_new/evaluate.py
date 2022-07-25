
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import sys
import numpy as np
import pandas as pd
from scipy import spatial
import tensorflow as tf

phys_gpus = tf.config.list_physical_devices('GPU')
for phys_g in phys_gpus:
    tf.config.experimental.set_memory_growth(phys_g, True)

from default_config.util import *
from tf2.usage.predictor import Predictor

params = masif_opts['ligand_site']
ligand_list = params['ligand_list']

ligand_coord_dir = params["ligand_coords_dir"]
precom_dir = params['masif_precomputation_dir']
binding_dir = '/data02/daniel/PUresNet/site_predictions'

pred = Predictor(ligand_model_path = '/home/daniel.monyak/software/masif/source/tf2/masif_ligand/l2/kerasModel/savedModel')

listDir = '/home/daniel.monyak/software/masif/data/masif_ligand/lists'
train_file = 'train_pdbs_sequence.npy'
val_file = 'val_pdbs_sequence.npy'
test_file = 'test_pdbs_sequence.npy'

train_list = np.char.add(np.load(os.path.join(listDir, train_file)).astype(str), '_')
val_list = np.char.add(np.load(os.path.join(listDir, val_file)).astype(str), '_')
test_list = np.char.add(np.load(os.path.join(listDir, test_file)).astype(str), '_')


pdb_list = []
true_pocket_list = []
pred_pocket_list = []
dataset_list = []
recall_list = []
precision_list = []
ligandIdx_true_list = []
true_pts_ligandIdx_pred_list = []
pred_pts_ligandIdx_pred_list = []
npoints_true_list = []
npoints_pred_list = []

BIG_pdb_list = []
BIG_dataset_list = []
BIG_n_pockets_true = []
BIG_n_pockets_pred = []
BIG_matched = []

columns = ['pdb_list', 'true_pocket_list', 'pred_pocket_list', 'dataset_list', 'recall_list', 'precision_list', 'ligandIdx_true_list', 'true_pts_ligandIdx_pred_list', 'pred_pts_ligandIdx_pred_list', 'npoints_true_list', 'npoints_pred_list']
BIG_columns = ['BIG_pdb_list', 'BIG_dataset_list', 'BIG_n_pockets_true', 'BIG_n_pockets_pred', 'BIG_matched']

outdir = 'results'
if not os.path.exists(outdir):
    os.mkdir(outdir)

dataset_dict = {'train' : train_list, 'test' : test_list, 'val' : val_list}

#for dataset in dataset_dict.keys():
for dataset in ['test', 'val', 'test']:
    data = dataset_dict[dataset]
    n_data = len(data)
    
    j = 0
    for i, pdb in enumerate(data):
        print(f'\n{i} of {n_data} {dataset} pdbs running...')
        print(pdb, "\n")
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
        n_pockets_true = len(all_ligand_types)
        
        pdb_dir = os.path.join(precom_dir, pdb)
        xyz_coords = Predictor.getXYZCoords(pdb_dir)
        tree = spatial.KDTree(xyz_coords)
        pred.loadData(pdb_dir)
        
        pp_true_list = []
        for lig_i in range(n_pockets_true):
            print(f'Pocket {lig_i}')
            
            ligand_coords = all_ligand_coords[lig_i]
            pocket_points_true = tree.query_ball_point(ligand_coords, 3.0)
            pocket_points_true = list(set([pp for p in pocket_points_true for pp in p]))
            
            if len(pocket_points_true) == 0:
                print(f'\tLigand has no pocket points...')
                continue

            pp_true_list.append(pocket_points_true)
        
        if len(pp_true_list) == 0:
            print('Zero true pockets...')
        
        ####################
        pdb_pnet_dir = os.path.join(binding_dir, pdb.rstrip("_"))
        files = os.listdir(pdb_pnet_dir)
        if len(files) == 0:
            print('Zero pockets were predicted...')
            BIG_pdb_list.append(pdb)
            BIG_dataset_list.append(dataset)
            BIG_n_pockets_true.append(n_pockets_true)
            BIG_n_pockets_pred.append(0)
            BIG_matched.append(0)
            continue

        n_pockets_pred = np.sum(np.char.endswith(files, '.txt'))
        
        #unmatched = 0
        matched = 0
        
        for pocket in range(n_pockets_pred):
            pnet_coords = np.loadtxt(os.path.join(pdb_pnet_dir, f'pocket{pocket}.txt'), dtype=float)
            pocket_points_pred = tree.query_ball_point(pnet_coords, 3.0)
            pocket_points_pred = list(set([pp for p in pocket_points_pred for pp in p]))
            
            npoints_pred = len(pocket_points_pred)
            if npoints_pred == 0:
                continue
            
            f1_highest = 0
            for ppt_idx, pocket_points_true in enumerate(pp_true_list):
                overlap = np.intersect1d(pocket_points_true, pocket_points_pred)
                if len(overlap) == 0:
                    continue
                    
                npoints_true = len(pocket_points_true)
                recall = len(overlap)/npoints_true
                precision = len(overlap)/npoints_pred
                f1 = 2*precision*recall/(precision+recall)
                if recall > 0.5 and f1 > f1_highest:
                    f1_highest = f1
                    ppt_idx_best = ppt_idx
            
            if f1_highest == 0:
                #unmatched += 1
                continue
                
            matched += 1
            
            pocket_points_true = pp_true_list[ppt_idx_best]
            del pp_true_list[ppt_idx_best]
            
            npoints_true = len(pocket_points_true)
            
            overlap = np.intersect1d(pocket_points_true, pocket_points_pred)
            recall = len(overlap)/npoints_true
            precision = len(overlap)/npoints_pred
            
            #try:
            true_pts_ligandIdx_pred = pred.predictLigandIdx(pred.getLigandX(pocket_points_true)).numpy()
            pred_pts_ligandIdx_pred = pred.predictLigandIdx(pred.getLigandX(pocket_points_pred)).numpy()
            #except:
            #    print('Something went wrong with ligand prediction...')
            #    continue
            
            ligand_true = all_ligand_types[ppt_idx_best]
            ligandIdx_true = ligand_list.index(ligand_true)
            
            ###############
            print(f'Predicted pocket: {pocket}')
            print(f'True pocket: {ppt_idx_best}')
            print(f'Recall: {recall}')
            print(f'Precision: {precision}')
            print(f'True ligand: {ligandIdx_true}')
            print(f'Prediction from true pocket points: {true_pts_ligandIdx_pred}')
            print(f'Prediction from predicted pocket points: {pred_pts_ligandIdx_pred}\n')

            pdb_list.append(pdb)
            true_pocket_list.append(ppt_idx_best)
            pred_pocket_list.append(pocket)
            dataset_list.append(dataset)
            recall_list.append(recall)
            precision_list.append(precision)
            ligandIdx_true_list.append(ligandIdx_true)
            true_pts_ligandIdx_pred_list.append(true_pts_ligandIdx_pred)
            pred_pts_ligandIdx_pred_list.append(pred_pts_ligandIdx_pred)
            npoints_true_list.append(npoints_true)
            npoints_pred_list.append(npoints_pred)
            
        #missed = len(pp_true_list)
        
        #print(f'{unmatched} unmatched predicted pockets')
        #print(f'{missed} missed true pockets')
        print(f'{matched} matched pockets of {n_pockets_true} true and {n_pockets_pred} predicted')
        
        BIG_pdb_list.append(pdb)
        BIG_dataset_list.append(dataset)
        BIG_n_pockets_true.append(n_pockets_true)
        BIG_n_pockets_pred.append(n_pockets_pred)
        BIG_matched.append(matched)
        
        print(f'\ni: {i}, (i > 0) and (i \% 50 == 0): {(i > 0) and (i % 50 == 0)}\n')
        if (i > 0) and (i % 50 == 0):
            results = pd.DataFrame(dict(zip([col.partition('_list')[0] for col in columns], [eval(col) for col in columns])))
            results.to_csv(os.path.join(outdir, 'results.csv'), index=False)

            BIG_results = pd.DataFrame(dict(zip([col.partition('_list')[0].partition('BIG_')[-1] for col in BIG_columns], [eval(col) for col in BIG_columns])))
            BIG_results.to_csv(os.path.join(outdir, 'BIG_results.csv'), index=False)


results = pd.DataFrame(dict(zip([col.partition('_list')[0] for col in columns], [eval(col) for col in columns])))
results.to_csv(os.path.join(outdir, 'results.csv'), index=False)

BIG_results = pd.DataFrame(dict(zip([col.partition('_list')[0].partition('BIG_')[-1] for col in BIG_columns], [eval(col) for col in BIG_columns])))
BIG_results.to_csv(os.path.join(outdir, 'BIG_results.csv'), index=False)

print('Finished!')
