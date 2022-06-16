import numpy as np
import os
from default_config.util import *
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
    
    self.n_predictions = n_predictions
    self.threshold = threshold
  
  def loadData(self, pdb_dir):
    self.input_feat = np.load(
        os.path.join(pdb_dir, "p1_input_feat.npy")
    )
    self.rho_coords = np.load(
        os.path.join(pdb_dir, "p1_rho_wrt_center.npy")
    )
    self.theta_coords = np.load(
        os.path.join(pdb_dir, "p1_theta_wrt_center.npy")
    )
    self.mask = np.load(
      os.path.join(pdb_dir, "p1_mask.npy")
    )
    self.data_dict = {'input_feat' : self.input_feat, 'rho_coords' : self.rho_coords,
                   'theta_coords' : self.theta_coords, 'mask' : self.mask}
  
  def getLigandSiteX(self):
    self.n_pockets = self.input_feat.shape[0]
    
    getFlatDataFromDict = lambda key : self.data_dict[key].flatten()
    flat_list = list(map(getFlatDataFromDict, data_order))
    return tf.RaggedTensor.from_tensor(
      tf.expand_dims(
        tf.concat(flat_list, axis=0),
        axis=0),
      ragged_rank = 1
    )
  
  def predictCoords(self, X):
    ligand_site_pred_list = []
    fullSamples = self.n_pockets // minPockets
    
    print('{} batches to run'.format(fullSamples))
    for i in range(fullSamples):
      if i % 10 == 0:
        done = 100.0 * i/fullSamples
        print('{} of {} batches completed. {}% done...'.format(i, fullSamples, done))
      sample = tf.expand_dims(tf.range(minPockets * i, minPockets * (i+1)), axis = 0)
      temp_pred = tf.squeeze(self.ligand_site_model(X, sample))
      ligand_site_pred_list.append(temp_pred)
    
    i = fullSamples
    garbage_idx =  self.n_pockets % minPockets
    valid = tf.range(minPockets * i, minPockets * i + garbage_idx)
    garbage = tf.zeros([minPockets - garbage_idx], dtype=tf.int32)
    sample = tf.expand_dims(tf.concat([valid,garbage], axis=0), axis=0)
    temp_pred = tf.squeeze(self.ligand_site_model(X, sample))
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
  
  
  def getLigandX(self, pocket_points):
    getDataFromDict = lambda key : tf.reshape(tf.gather(self.data_dict[key], pocket_points, axis = 0), [-1])
    flat_list = list(map(getDataFromDict, data_order))
    return tf.RaggedTensor.from_tensor(
      tf.expand_dims(
        tf.concat(flat_list, axis=0),
        axis=0),
      ragged_rank = 1
    )
  
  def predictLigandIdx(self, X):
    ligand_pred_list = []
    for i in range(self.n_predictions):
      temp_pred = tf.squeeze(self.ligand_model(X))
      ligand_pred_list.append(temp_pred)
    
    ligand_preds = tf.stack(ligand_pred_list, axis=0)
    ligand_preds_mean = np.mean(ligand_preds, axis=0)
    return ligand_preds_mean.argmax()
  
  def predict(self, pdb_dir):
    self.loadData(pdb_dir)
    
    ligand_site_X = self.getLigandSiteX()
    pocket_points = self.predictCoords(ligand_site_X)
    xyz_coords = self.getXYZCoords(pdb_dir)
    coords_list = xyz_coords[pocket_points]
    
    ligand_X = self.getLigandX(pocket_points)
    ligandIdx_pred = self.predictLigandIdx(ligand_X)
    ligand_pred = ligand_list[ligandIdx_pred]
    
    return (ligand_pred, coord_list)
