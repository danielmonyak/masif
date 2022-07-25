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
from tf2.LSResNet.LSResNet import LSResNet
from tf2.LSResNet.predict import predict



LSRN_threshold = 0.4




params = masif_opts['LSResNet']
ligand_list = params['ligand_list']

ligand_coord_dir = params["ligand_coords_dir"]
precom_dir = params['masif_precomputation_dir']
binding_dir = '/data02/daniel/PUresNet/site_predictions'

pred = Predictor()

LSRN_model = LSResNet(
    params["max_distance"],
    feat_mask=params["feat_mask"],
    n_thetas=4,
    n_rhos=3,
    n_rotations=4,
    extra_conv_layers = False
)
ckpPath = 'kerasModel/ckp'
load_status = LSRN_model.load_weights(ckpPath)
load_status.expect_partial()

listDir = '/home/daniel.monyak/software/masif/data/masif_ligand/lists'
train_file = 'train_pdbs_sequence.npy'
val_file = 'val_pdbs_sequence.npy'
test_file = 'test_pdbs_sequence.npy'

train_list = np.char.add(np.load(os.path.join(listDir, train_file)).astype(str), '_')
val_list = np.char.add(np.load(os.path.join(listDir, val_file)).astype(str), '_')
test_list = np.char.add(np.load(os.path.join(listDir, test_file)).astype(str), '_')


pdb_list = []
dataset_list = []
recall_list = []
precision_list = []
npoints_true_list = []
npoints_pred_list = []

BIG_pdb_list = []
BIG_dataset_list = []
BIG_n_pockets_true = []
BIG_n_pockets_pred = []
BIG_matched = []

columns = ['pdb_list', 'dataset_list', 'recall_list', 'precision_list', 'npoints_true_list', 'npoints_pred_list']
BIG_columns = ['BIG_pdb_list', 'BIG_dataset_list', 'BIG_n_pockets_true', 'BIG_n_pockets_pred', 'BIG_matched']

outdir = 'results'
if not os.path.exists(outdir):
    os.mkdir(outdir)

dataset_dict = {'train' : train_list, 'test' : test_list, 'val' : val_list}

#for dataset in dataset_dict.keys():
for dataset in ['test', 'val', 'train']:
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

        PU_RN_pp_pred = []
        n_txt_files = np.sum(np.char.endswith(files, '.txt'))
        if n_txt_files > 0:
            for pocket in range(n_txt_files):
                coords = np.loadtxt(os.path.join(pdb_pnet_dir, f'pocket{pocket}.txt'), dtype=float)
                pocket_points_pred = tree.query_ball_point(coords, 3.0)
                pocket_points_pred = list(set([pp for p in pocket_points_pred for pp in p]))
                if len(pocket_points_pred) > 0:
                    PU_RN_pp_pred.append(pocket_points_pred)
        
        ####################
        LS_RN_pocket_coords = predict(LSRN_model, pdb, threshold=LSRN_threshold)
        n_pockets_pred = len(LS_RN_pocket_coords)
        
        ##########
        if n_pockets_pred == 0:
            print('Zero pockets were predicted...')
            BIG_pdb_list.append(pdb)
            BIG_dataset_list.append(dataset)
            BIG_n_pockets_true.append(n_pockets_true)
            BIG_n_pockets_pred.append(0)
            BIG_matched.append(0)
            continue
        
        ##########
        LS_RN_pp_pred = []
        for coords in LS_RN_pocket_coords:
            pocket_points_pred = tree.query_ball_point(coords, 3.0)
            pocket_points_pred = list(set([pp for p in pocket_points_pred for pp in p]))
            if len(pocket_points_pred) > 0:
                LS_RN_pp_pred.append(pocket_points_pred)
        
        ###########################################
        final_pp_pred_list = []
        for i, LS_pp in enumerate(LS_RN_pp_pred):
            matched_pred_pocket = -1
            for i, PU_pp in enumerate(PU_RN_pp_pred):
                print(i)
                overlap = np.intersect1d(PU_pp, LS_pp)
                recall_1 = len(overlap)/len(PU_pp)
                recall_2 = len(overlap)/len(LS_pp)
                if (recall_1 > 0.25) or (recall_2 > 0.25):
                    matched_pred_pocket = i
                    final_pp_pred_list.append(overlap)
                    break
            if matched_pred_pocket == -1:
                final_pp_pred_list.append(LS_pp)
            else:
                del PU_RN_pp_pred[matched_pred_pocket]
        
        ###########################################
        matched = 0
        for pocket_points_pred in final_pp_pred_list:
            npoints_pred = len(pocket_points_pred)
            
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
                continue
                
            matched += 1
            
            pocket_points_true = pp_true_list[ppt_idx_best]
            del pp_true_list[ppt_idx_best]
            
            npoints_true = len(pocket_points_true)
            
            overlap = np.intersect1d(pocket_points_true, pocket_points_pred)
            recall = len(overlap)/npoints_true
            precision = len(overlap)/npoints_pred
            
            ###############
            print(f'Predicted pocket: {pocket}')
            print(f'True pocket: {ppt_idx_best}')
            print(f'Recall: {recall}')
            print(f'Precision: {precision}')
            
            pdb_list.append(pdb)
            dataset_list.append(dataset)
            recall_list.append(recall)
            precision_list.append(precision)
            npoints_true_list.append(npoints_true)
            npoints_pred_list.append(npoints_pred)
            
        print(f'{matched} matched pockets of {n_pockets_true} true and {n_pockets_pred} predicted')
        
        BIG_pdb_list.append(pdb)
        BIG_dataset_list.append(dataset)
        BIG_n_pockets_true.append(n_pockets_true)
        BIG_n_pockets_pred.append(n_pockets_pred)
        BIG_matched.append(matched)
        
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
