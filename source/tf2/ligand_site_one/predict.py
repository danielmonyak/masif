import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import sys
import numpy as np
import tensorflow as tf
import tfbio.data

phys_gpus = tf.config.list_physical_devices('GPU')
for phys_g in phys_gpus:
    tf.config.experimental.set_memory_growth(phys_g, True)

from default_config.masif_opts import masif_opts
from tf2.ligand_site_one.get_data import get_data

from skimage.segmentation import clear_border
from skimage.morphology import closing
from skimage.measure import label

params = masif_opts["LSResNet"]
ligand_coord_dir = params["ligand_coords_dir"]

#possible_test_pdbs = ['2VRB_AB_', '1FCD_AC_', '1FNN_A_', '1RI4_A_', '4PGH_AB_']
#possible_train_pdbs = ['4X7G_A_', '4RLR_A_', '3OWC_A_', '3SC6_A_', '1TU9_A_']


def predict(model, pdb, threshold=0.5, min_size=50, make_y=False, mode='pdb_id'):
    data = get_data(pdb.rstrip('_'), training=False, make_y=make_y, mode=mode)
    if data is None:
        print('Data couldn\'t be retrieved')
        return None

    X, y, _ = data
    
    mydir = os.path.join(params["masif_precomputation_dir"], pdb.rstrip('_') + '_')
    X_coords = np.load(os.path.join(mydir, "p1_X.npy"))
    Y_coords = np.load(os.path.join(mydir, "p1_Y.npy"))
    Z_coords = np.load(os.path.join(mydir, "p1_Z.npy"))
    xyz_coords = np.vstack([X_coords, Y_coords, Z_coords]).T

    centroid = xyz_coords.mean(axis=0)
    xyz_coords -= centroid
    
    probs = tf.sigmoid(model.predict(X)).numpy()

    resolution = 1. / params['scale']
    density = tfbio.data.make_grid(xyz_coords, probs[0], max_dist=params['max_dist'], grid_resolution=resolution)

    origin = (centroid - params['max_dist'])
    step = np.array([1.0 / params['scale']] * 3)
    
    voxel_size = (1 / params['scale']) ** 3
    bw = closing((density[0] > threshold).any(axis=-1))
    cleared = clear_border(bw)

    label_image, num_labels = label(cleared, return_num=True)
    for i in range(1, num_labels + 1):
        pocket_idx = (label_image == i)
        pocket_size = pocket_idx.sum() * voxel_size
        if pocket_size < min_size:
            label_image[np.where(pocket_idx)] = 0

    pockets = label_image
    
    #pockets = np.squeeze(density) > threshold

    pocket_label_arr = np.unique(pockets)
    ligand_coords_arr = []
    
    for pocket_label in pocket_label_arr[pocket_label_arr > 0]:
        indices = np.argwhere(pockets == pocket_label).astype('float32')
        indices *= step
        indices += origin
        ligand_coords_arr.append(indices)

    return ligand_coords_arr
