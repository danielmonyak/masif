from default_config.util import *
from sklearn.cluster import KMeans
from sklearn.metrics import balanced_accuracy
from scipy import spatial
from predictor import Predictor

params = masif_opts['ligand']
ligand_list = params['ligand_list']

ligand_coord_dir = params["ligand_coords_dir"]
precom_dir = '/data02/daniel/masif/data_preparation/04a-precomputation_12A/precomputation'
ligand_model_path = '/home/daniel.monyak/software/masif/source/tf2/kerasModel/savedModel'
ligand_site_ckp_path = '/home/daniel.monyak/software/masif/source/ligand_site/kerasModel/ckp'

pred = Predictor(ligand_model_path, ligand_site_ckp_path)

listDir = '/home/daniel.monyak/software/masif/data/masif_ligand/lists'
fileName = 'test.npy'
test_list = np.load(os.path.join(listDir, fileName))

f1_scores = []
lig_true = []
lig_pred = []

n_test = 20
for i, pdb in enumerate(test_list):
    if i == 20:
        break
    
    print('{} of {} test pdbs running...'.format(i, n_test))
    pdb_dir = os.path.join(precom_dir, pdb)
    ligandIdx_pred, pocket_points_pred = pred.predictRaw(pdb_dir)
    xyz_coords = pred.getXYZCoords(pdb_dir)
    
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
    pocket_points_true = np.array(set([pp for p in pocket_points_true for pp in p]))
    
    overlap = np.intersect1d(pocket_points_true, pocket_points_pred)
    recall = overlap.shape[0]/pocket_points_true.shape[0]
    precision = overlap.shape[0]/pocket_points_pred.shape[0]
    # Is this F1 score???
    ########################
    f1 = (recall + precision)/2
    f1_scores.append(f1)
    ########################
    ligandIdx_true = all_ligand_types[0]
    y_true.append(ligand_list.idxOf(ligandIdx_true))
    y_pred.append(ligandIdx_pred)

#f1_scores = np.array(f1_scores)
#y_true = np.array(y_true)
#y_pred = np.array(y_pred)

# order of args?
bal_acc = balanced_accuracy(y_true, y_pred)

confusion_matrix(y_true, y_pred)
