import os
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from scipy import spatial
from default_config.util import *
from tf2.usage.predictor import Predictor

continue_running = False
outdir = 'results'

params = masif_opts['ligand']
ligand_list = params['ligand_list']

ligand_coord_dir = params["ligand_coords_dir"]
precom_dir = '/data02/daniel/masif/data_preparation/04a-precomputation_12A/precomputation'
ligand_model_path = '/home/daniel.monyak/software/masif/source/tf2/masif_ligand/kerasModel/savedModel'
ligand_site_ckp_path = '/home/daniel.monyak/software/masif/source/tf2/ligand_site/kerasModel/ckp'

thresh = 0.5
pred = Predictor(ligand_model_path, ligand_site_ckp_path, ligand_threshold = thresh)

listDir = '/home/daniel.monyak/software/masif/data/masif_ligand/lists'
test_file = 'test_pdbs_sequence.npy'
train_file = 'train_pdbs_sequence.npy'

test_list = np.char.add(
        np.load(
                os.path.join(listDir, test_file)
        ).astype(str), '_')

train_list = np.char.add(
        np.load(
            os.path.join(listDir, train_file)
        ).astype(str), '_')

if not os.path.exists(outdir):
    os.mkdir(outdir)

pdb_file = os.path.join(outdir, 'pdbs.txt')
recall_file = os.path.join(outdir, 'recall.txt')
precision_file = os.path.join(outdir, 'precision.txt')
lig_true_file = os.path.join(outdir, 'lig_true.txt')
lig_pred_file = os.path.join(outdir, 'lig_pred.txt')

if continue_running:
    pdbs_done = np.loadtxt(pdb_file, dtype=str)
    pdbs_left = test_list[~np.isin(test_list, pdbs_done)]
else:
    pdbs_left = test_list
    for fi in [pdb_file, recall_file, precision_file, lig_true_file, lig_pred_file]:
        with open(fi, 'w') as f:
            pass

#break_i = 2

n_train = 5
n_test = len(pdbs_left)

dev = '/GPU:0'
with tf.device(dev):
    for i, pdb in enumerate(pdbs_left):
        print('{} of {} test pdbs running...'.format(i, n_test))
        try:
            pdb_dir = os.path.join(precom_dir, pdb)
            ligandIdx_pred, pocket_points_pred = pred.predictRaw(pdb_dir)
            xyz_coords = Predictor.getXYZCoords(pdb_dir)
            
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
        
            ligand_coords = all_ligand_coords[0]
            tree = spatial.KDTree(xyz_coords)
            pocket_points_true = tree.query_ball_point(ligand_coords, 3.0)
            pocket_points_true = list(set([pp for p in pocket_points_true for pp in p]))
            
            overlap = np.intersect1d(pocket_points_true, pocket_points_pred)
            recall = len(overlap)/len(pocket_points_true)
            precision = len(overlap)/len(pocket_points_pred)
            
            ligand_true = all_ligand_types[0]
            ligandIdx_true = ligand_list.index(ligand_true)
        except:
            continue
        
        with open(pdb_file, 'a') as f:
            f.write(str(pdb) + '\n')
        
        with open(recall_file, 'a') as f:
            f.write(str(recall) + '\n')
        with open(precision_file, 'a') as f:
            f.write(str(precision) + '\n')
        
        with open(lig_true_file, 'a') as f:
            f.write(str(ligandIdx_true) + '\n')
        with open(lig_pred_file, 'a') as f:
            f.write(str(ligandIdx_pred) + '\n')
