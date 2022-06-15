import numpy as np
import os
from default_config.masif_opts import masif_opts
import tensorflow as tf
from ligand_site.MaSIF_ligand_site import MaSIF_ligand_site

params = masif_opts['ligand']
ligand_list = params['ligand_list']
minPockets = params['minPockets']

class Predictor:
  def getLigandSiteModel(self, ligand_site_ckp_path):
    ligand_site_model = MaSIF_ligand_site(
      params["max_distance"],
      params["n_classes"],
      feat_mask=params["feat_mask"],
      keep_prob = 1.0
    )
    ligand_site_model.compile(optimizer = ligand_site_model.opt,
      loss = ligand_site_model.loss_fn,
      metrics=['accuracy']
    )
    ligand_site_model.load_weights(ligand_site_ckp_path)
    return ligand_site_model
    
  def __init__(self, ligand_model_path, ligand_site_ckp_path, n_predictions = 100, threshold = 0.5):
    self.ligand_model = tf.keras.models.load_model(ligand_model_path)
    #self.ligand_site_model = tf.keras.models.load_model(ligand_site_model_path)
    
    self.ligand_site_model = self.getLigandSiteModel(ligand_site_ckp_path)
    
    
    self.key_list = ['input_feat', 'rho_coords', 'theta_coords', 'mask']
    self.getDataFromDict = lambda key : self.key_list[key]
    
    self.n_predictions = n_predictions
    self.threshold = threshold
  
  def getData(self, pdb_dir):
    input_feat = np.load(
        os.path.join(pdb_dir, "p1_input_feat.npy")
    )
    rho_coords = np.load(
        os.path.join(pdb_dir, "p1_rho_wrt_center.npy")
    )
    theta_coords = np.load(
        os.path.join(pdb_dir, "p1_theta_wrt_center.npy")
    )
    mask = np.load(
      os.path.join(pdb_dir, "p1_mask.npy")
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
    
    print('fullSamples: {}'.format(fullSamples))
    for i in range(fullSamples + 1):
      print(i)
      sample = tf.expand_dims(tf.range(minPockets * i, minPockets * (i+1)), axis = 0)
      temp_pred = tf.squeeze(self.ligand_site_model(X, sample))
      ligand_site_pred_list.append(temp_pred)
    
    i = fullSamples
    garbage_idx =  self.n_pockets % minPockets
    valid = tf.range(minPockets * i, minPockets * i + garbage_idx)
    garbage = tf.zeros([minPockets - garbage_idx], dtype=tf.int32)
    sample = tf.expand_dims(tf.concat([valid,garbage], axis=0), axis=0)
    temp_pred = tf.squeeze(self.ligand_site_model(X, sample)
    ligand_site_pred_list.append(temp_pred)
    
    ligand_site_preds = tf.concat(ligand_site_pred_list, axis = 0)
    ligand_site_preds = ligand_site_preds[:minPockets * i + garbage_idx]
    coords_list = tf.where(ligand_site_preds > self.threshold)
    return tf.squeeze(coords_list)
  
  def getXYZCoords(self, pdb_dir):
    X = np.load(os.path.join(pdb_dir, "p1_X.npy"))
    Y = np.load(os.path.join(pdb_dir, "p1_Y.npy"))
    Z = np.load(os.path.join(pdb_dir, "p1_Z.npy"))
    xyz_coords = np.vstack([X, Y, Z]).T
    return xyz_coords
  
  def predict(self, pdb_dir):
    X = self.getData(pdb_dir)
    
    ligandIdx_pred = self.predictLigandIdx(X)
    ligand_pred = ligand_list[ligandIdx_pred]
    
    coords_list = self.predictCoords(X)
    xyz_coords = self.getXYZCoords(pdb_dir)
    coord_list = xyz_coords[coords_list]
    
    return (ligand_pred, coord_list)
