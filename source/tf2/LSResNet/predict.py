import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import sys
import numpy as np
import tensorflow as tf

from skimage.segmentation import clear_border
from skimage.morphology import closing
from skimage.measure import label

phys_gpus = tf.config.list_physical_devices('GPU')
for phys_g in phys_gpus:
    tf.config.experimental.set_memory_growth(phys_g, True)

from default_config.util import *
from tf2.LSResNet.get_data import get_data

params = masif_opts["LSResNet"]
ligand_coord_dir = params["ligand_coords_dir"]

def predict(model, pdb, threshold=0.5, min_size=50):
    data = get_data(pdb.rstrip('_'), training=False)
    if data is None:
        print('Data couldn\'t be retrieved')
        return None
    
    X, y, centroid = data
    prot_coords = np.squeeze(X[1])

    density = tf.sigmoid(model.predict(X)).numpy()

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

    pocket_label_arr = np.unique(pockets)
    ligand_coords_arr = []
    
    for pocket_label in pocket_label_arr[pocket_label_arr > 0]:
        indices = np.argwhere(pockets == pocket_label).astype('float32')
        indices *= step
        indices += origin
        ligand_coords_arr.append(indices)

    return ligand_coords_arr
        
'''y_pred = density > threshold
mask = y_pred & y.astype(bool)
overlap = np.sum(mask)
recall = overlap/np.sum(y)
precision = overlap/np.sum(y_pred)

print(f'Recall: {recall}')
print(f'Precision: {precision}')'''
