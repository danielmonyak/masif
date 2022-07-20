import os
import numpy as np
from default_config.util import *
#from tf2.ligand_site_one.MaSIF_ligand_site_one import MaSIF_ligand_site
from tf2.ligand_site_one.sep_layers.MaSIF_ligand_site_one import MaSIF_ligand_site
from time import process_time

params = masif_opts['ligand']
ligand_site_params = masif_opts['ligand_site']
ligand_list = params['ligand_list']
minPockets = params['minPockets']
with tf.device('/GPU:2'):
  gen_sample = tf.expand_dims(tf.range(minPockets), axis = 0)

class Predictor:
  def getLigandSiteModel(self, ligand_site_ckp_path):
    model = MaSIF_ligand_site(
        params["max_distance"],
        feat_mask=params["feat_mask"],
        n_thetas=4,
        n_rhos=3,
        learning_rate = 1e-4,
        n_rotations=4,
        reg_val = 0
    )
    from_logits = model.loss_fn.get_config()['from_logits']
    binAcc = tf.keras.metrics.BinaryAccuracy(threshold = (not from_logits) * 0.5)
    model.compile(optimizer = model.opt,
      loss = model.loss_fn,
      metrics=[binAcc]
    )
    model.load_weights(ligand_site_ckp_path)
    return model
    
  def __init__(self, ligand_model_path = None, ligand_site_ckp_path = None, ligand_site_model_path = None, n_predictions = 100, threshold = 0.5, ligand_threshold = 0):
    # Load MaSIF_ligand and MaSIF_ligand_site models
    # MaSIF_ligand_site model comes from saved checkpoint
    if not ligand_model_path is None:
      self.ligand_model = tf.keras.models.load_model(ligand_model_path)
    if not ligand_site_ckp_path is None:
      self.ligand_site_model = self.getLigandSiteModel(ligand_site_ckp_path)
    if not ligand_site_model_path is None:
      self.ligand_site_model = tf.keras.models.load_model(ligand_site_model_path)
    
    self.n_predictions = n_predictions
    self.threshold = threshold
    self.ligand_threshold = ligand_threshold
  
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
    self.mask = np.expand_dims(np.load(
      os.path.join(pdb_dir, "p1_mask.npy")
    ), axis=-1)
    
    self.n_samples = self.mask.shape[0]
    self.n_verts = self.mask.shape[1]
    
    self.indices = pad_indices(np.load(os.path.join(pdb_dir, 'p1_list_indices.npy'), encoding="latin1", allow_pickle = True), self.n_verts).astype(np.int32)
    
    self.data_dict = {'input_feat' : self.input_feat, 'rho_coords' : self.rho_coords,
                   'theta_coords' : self.theta_coords, 'mask' : self.mask}
  
  def getLigandSiteProbs(self):
    data_tsrs = tuple(np.expand_dims(tsr, axis=0) for tsr in [input_feat, rho_wrt_center, theta_wrt_center, mask])
    indices = np.expand_dims(indices, axis=0)
    X = (data_tsrs, indices)
    return self.ligand_site_model.predict(X)
  
  # Wrapper function for getLigandSiteProbs
  def predictPocketPoints(self, threshold = None):
    ligand_site_probs = self.getLigandSiteProbs()
    if threshold is None:
      threshold = self.threshold
    return (ligand_site_probs > threshold).nonzero()[0]
  
  # Get geometric coordinates of PDB
  def getXYZCoords(pdb_dir):
    X = np.load(os.path.join(pdb_dir, "p1_X.npy"))
    Y = np.load(os.path.join(pdb_dir, "p1_Y.npy"))
    Z = np.load(os.path.join(pdb_dir, "p1_Z.npy"))
    xyz_coords = np.vstack([X, Y, Z]).T
    return xyz_coords
  
  # Get input to MaSIF_ligand, using pocket_points
  def getLigandX(self, pocket_points):
    getDataFromDict = lambda key : self.data_dict[key][pocket_points].flatten()
    flat_list = list(map(getDataFromDict, data_order))
    return tf.RaggedTensor.from_tensor(
      tf.expand_dims(
        tf.concat(flat_list, axis=0),
        axis=0),
      ragged_rank = 1
    )
  
  # Run MaSIF_ligand on pdb, return prob of each ligand (based on ligand_list in defaul_config/util.py)
  def predictLigandProbs(self, X, threshold=None):
    if threshold is None:
      threshold = self.ligand_threshold
    
    ligand_prob_list = []
    for i in range(self.n_predictions):
      temp_prob = tf.sigmoid(tf.squeeze(self.ligand_model(X)))
      if tf.reduce_max(temp_prob) > self.ligand_threshold:
        ligand_prob_list.append(temp_prob)
    
    ligand_probs = tf.stack(ligand_prob_list, axis=0)
    return tf.reduce_mean(ligand_probs, axis=0)
  
  def predictLigandIdx(self, X, threshold=None):
    if threshold is None:
      threshold = self.ligand_threshold
      
    ligand_probs_mean = self.predictLigandProbs(X, threshold)
    return tf.math.argmax(ligand_probs_mean)
    
  
  # Run input through both models, return index of ligand prediction, pocket_points
  def predictRaw(self, pdb_dir):
    self.loadData(pdb_dir)
    
    pocket_points = self.predictPocketPoints()
    
    ligand_X = self.getLigandX(pocket_points)
    ligandIdx_pred = self.predictLigandIdx(ligand_X)

    return (ligandIdx_pred, pocket_points)
  
  # Call predictRaw and return name of predicted ligand, coordinates of pocket_points
  def predict(self, pdb_dir):
    ligandIdx_pred, pocket_points = self.predictRaw(pdb_dir)

    xyz_coords = Predictor.getXYZCoords(pdb_dir)
    coords_list = xyz_coords[pocket_points]
    ligand_pred = ligand_list[ligandIdx_pred]
    
    return (ligand_pred, coords_list)
