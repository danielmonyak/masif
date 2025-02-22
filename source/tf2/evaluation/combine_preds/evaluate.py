# Run LSResNet on all proteins in test dataset, use PUResNet predictions to validate

## Whether or not to make predictions for every pocket with MaSIF-ligand - takes more time
predict_ligand = False

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

import default_config.util as util
from default_config.masif_opts import masif_opts
from tf2.predictor import Predictor
from tf2.LSResNet.LSResNet import LSResNet
from tf2.LSResNet.predict import predict

##############################################
##############################################
LSRN_threshold = 0.3

##############################################
##############################################

params = masif_opts['LSResNet']
ligand_list = params['ligand_list']

ligand_coord_dir = params["ligand_coords_dir"]
precom_dir = params['masif_precomputation_dir']
binding_dir = '/data02/daniel/PUresNet/site_predictions'

if predict_ligand:
    pred = Predictor(ligand_model_path = '/home/daniel.monyak/software/masif/source/tf2/masif_ligand/l2/kerasModel/savedModel')

LSRN_model = LSResNet(
    params["max_distance"],
    feat_mask=params["feat_mask"],
    n_thetas=4,
    n_rhos=3,
    n_rotations=4,
    extra_conv_layers = False
)
ckpPath = '/home/daniel.monyak/software/masif/source/tf2/LSResNet/batch/kerasModel/ckp'
load_status = LSRN_model.load_weights(ckpPath)
load_status.expect_partial()

listDir = '/home/daniel.monyak/software/masif/data/masif_ligand/newPDBs/lists'
train_file = 'train_reg.npy'
val_file = 'val_reg.npy'
test_file = 'test_reg.npy'

train_list = np.load(os.path.join(listDir, train_file)).astype(str)
val_list = np.load(os.path.join(listDir, val_file)).astype(str)
test_list = np.load(os.path.join(listDir, test_file)).astype(str)


pdb_list = []
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

if predict_ligand:
    columns = ['pdb_list', 'dataset_list', 'recall_list', 'precision_list', 'ligandIdx_true_list', 'true_pts_ligandIdx_pred_list', 'pred_pts_ligandIdx_pred_list', 'npoints_true_list', 'npoints_pred_list']
else:
    columns = ['pdb_list', 'dataset_list', 'recall_list', 'precision_list', 'npoints_true_list', 'npoints_pred_list']

BIG_columns = ['BIG_pdb_list', 'BIG_dataset_list', 'BIG_n_pockets_true', 'BIG_n_pockets_pred', 'BIG_matched']

outdir = 'results'
if not os.path.exists(outdir):
    os.mkdir(outdir)

dataset_dict = {'train' : train_list, 'test' : test_list, 'val' : val_list}

#for dataset in dataset_dict.keys():
for dataset in ['test']:
    data = dataset_dict[dataset]
    n_data = len(data)
    
    for i, pdb in enumerate(data):
        print(f'\n{i} of {n_data} {dataset} pdbs running...')
        print(pdb, "\n")
        try:
            all_ligand_coords = np.load(
                os.path.join(
                    ligand_coord_dir, "{}_ligand_coords.npy".format(pdb.split("_")[0])
                ), allow_pickle=True, encoding='latin1'
            )
            all_ligand_types = np.load(
                os.path.join(
                    ligand_coord_dir, "{}_ligand_types.npy".format(pdb.split("_")[0])
                )
            ).astype(str)
        except:
            continue
        
        ####################
        
        pdb_dir = os.path.join(precom_dir, pdb)
        try:
            xyz_coords = Predictor.getXYZCoords(pdb_dir)
            tree = spatial.KDTree(xyz_coords)
        except:
            continue
            
        pp_true_list = []
        lig_true_list = []
        for lig_i, structure_ligand in enumerate(all_ligand_types):
            if not structure_ligand in ligand_list:
                continue
            
            ligand_coords = all_ligand_coords[lig_i]
            pocket_points_true = tree.query_ball_point(ligand_coords, 3.0)
            pocket_points_true = list(set([pp for p in pocket_points_true for pp in p]))
            
            if len(pocket_points_true) == 0:
                continue

            pp_true_list.append(pocket_points_true)
            lig_true_list.append(structure_ligand)
        
        n_pockets_true = len(pp_true_list)
        
        if n_pockets_true == 0:
            print('Zero true pockets...')
        
        ####################
        pdb_pnet_dir = os.path.join(binding_dir, pdb.rstrip("_"))
        files = os.listdir(pdb_pnet_dir)
        PU_RN_pp_pred = []
        if len(files) > 0:
            for pocket in range(np.sum(np.char.endswith(files, '.txt'))):
                coords = np.loadtxt(os.path.join(pdb_pnet_dir, f'pocket{pocket}.txt'), dtype=float)
                pocket_points_pred = tree.query_ball_point(coords, 3.0)
                pocket_points_pred = list(set([pp for p in pocket_points_pred for pp in p]))
                if len(pocket_points_pred) > 0:
                    PU_RN_pp_pred.append(pocket_points_pred)
        
        ####################
        LS_RN_pocket_coords = predict(LSRN_model, pdb, threshold=LSRN_threshold)
        if LS_RN_pocket_coords is None:
            n_pockets_pred = 0
        else:
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
        for LS_pp in LS_RN_pp_pred:
            matched_pred_pocket = -1
            for i, PU_pp in enumerate(PU_RN_pp_pred):
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
            
            if predict_ligand:
                try:
                    true_pts_ligandIdx_pred = pred.predictLigandIdx(pred.getLigandX(pocket_points_true)).numpy()
                    pred_pts_ligandIdx_pred = pred.predictLigandIdx(pred.getLigandX(pocket_points_pred)).numpy()
                except:
                    print('Something went wrong with ligand prediction...')
                    continue
                ligand_true = lig_true_list[ppt_idx_best]
                ligandIdx_true = ligand_list.index(ligand_true)
            
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
            
            if predict_ligand:
                print(f'True ligand: {ligandIdx_true}')
                print(f'Prediction from true pocket points: {true_pts_ligandIdx_pred}')
                print(f'Prediction from predicted pocket points: {pred_pts_ligandIdx_pred}\n')
                ligandIdx_true_list.append(ligandIdx_true)
                true_pts_ligandIdx_pred_list.append(true_pts_ligandIdx_pred)
                pred_pts_ligandIdx_pred_list.append(pred_pts_ligandIdx_pred)
            
        print(f'{matched} matched pockets of {n_pockets_true} true and {n_pockets_pred} predicted')
        
        BIG_pdb_list.append(pdb)
        BIG_dataset_list.append(dataset)
        BIG_n_pockets_true.append(n_pockets_true)
        BIG_n_pockets_pred.append(n_pockets_pred)
        BIG_matched.append(matched)
        

results = pd.DataFrame(dict(zip([col.partition('_list')[0] for col in columns], [eval(col) for col in columns])))
results.to_csv(os.path.join(outdir, 'results.csv'), index=False)

BIG_results = pd.DataFrame(dict(zip([col.partition('_list')[0].partition('BIG_')[-1] for col in BIG_columns], [eval(col) for col in BIG_columns])))
BIG_results.to_csv(os.path.join(outdir, 'BIG_results.csv'), index=False)

print('Finished!')
