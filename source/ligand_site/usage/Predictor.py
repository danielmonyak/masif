import numpy as np
import os
from default_config.masif_opts import masif_opts
import tensorflow as tf

params = masif_opts['ligand']
ligand_list = params['ligand_list']
minPockets = params['minPockets']

class Predictor:
  def __init__(self, ligand_model_path, ligand_site_model_path, n_predictions = 100, threshold = 0.5):
    self.ligand_model = tf.keras.models.load_model(ligand_model_path)
    self.ligand_site_model = tf.keras.models.load_model(ligand_site_model_path)
    
    self.key_list = ['input_feat', 'rho_coords', 'theta_coords', 'mask']
    self.getDataFromDict = lambda key : self.key_list[key]
    
    self.n_predictions = n_predictions
    self.threshold = threshold
  
  def getData(self, precom_dir):
    input_feat = np.load(
        os.path.join(precom_dir, "p1_input_feat.npy")
    )
    rho_coords = np.load(
        os.path.join(precom_dir, "p1_rho_wrt_center.npy")
    )
    theta_coords = np.load(
        os.path.join(precom_dir, "p1_theta_wrt_center.npy")
    )
    mask = np.load(
      os.path.join(precom_dir, "p1_mask.npy")
    )
    self.n_pockets = input_feat.shape[0]

    data_dict = {'input_feat' : input_feat.flatten(), 'rho_coords' : rho_coords.flatten(),
                 'theta_coords' : theta_coords.flatten(), 'mask' : mask.flatten()}
    getDataFromDict = lambda key : data_dict[key]
    flat_list = list(map(getDataFromDict, self.key_list))
    return tf.RaggedTensor.from_tensor(
      tf.expand_dims(
        tf.concat(flat_list, axis=0),
        axis=0),
      ragged_rank = 1
    )
  
  def predictLigandIdx(self, X):
    ligand_pred_list = []
    for i in range(n_predictions):
      ligand_pred_list.append(self.ligand_model(X))
    
    ligand_preds = np.vstack(ligand_pred_list)
    ligand_preds_mean = np.mean(ligand_preds, axis=0)
    return ligand_preds_mean.argmax()
  
  def predictCoords(self, X):
    ligand_site_pred_list = []
    fullSamples = self.n_pockets // minPockets
    garbage_idx = self.n_pockets % minPockets
    for i in range(fullSamples + 1):
      sample = tf.range(minPockets * i, minPockets * (i+1))
      if i == fullSamples:
        sample[garbage_idx:] = 0
      ligand_site_pred_list.append(self.ligand_site_model(X, sample = sample))

    ligand_site_preds_raw = np.vstack(ligand_site_pred_list)
    ligand_site_preds = ligand_site_preds_raw[:garbage_idx]
    coords_list = np.where(ligand_site_preds > self.threshold)
    return coords_list
  
  def getXYZCoords(self, precom_dir):
    X = np.load(os.path.join(precom_dir, pdb + "_", "p1_X.npy"))
    Y = np.load(os.path.join(precom_dir, pdb + "_", "p1_Y.npy"))
    Z = np.load(os.path.join(precom_dir, pdb + "_", "p1_Z.npy"))
    xyz_coords = np.vstack([X, Y, Z]).T
    return xyz_coords
  
  def predict(self, precom_dir):
    X = self.getData(precom_dir)
    
    ligandIdx_pred = self.predictLigandIdx(X)
    ligand_pred = ligand_list[ligandIdx_pred]
    
    coords_list = self.predictCoords(X)
    xyz_coords = self.getXYZCoords(precom_dir)
    coord_list = xyz_coords[coords_list]
    
    return (ligand_pred, coord_list)
