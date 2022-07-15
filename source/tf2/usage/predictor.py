import os
import numpy as np
from default_config.util import *
from tf2.ligand_site.MaSIF_ligand_site import MaSIF_ligand_site
from time import process_time

params = masif_opts['ligand']
ligand_list = params['ligand_list']
minPockets = params['minPockets']
with tf.device('/GPU:2'):
  gen_sample = tf.expand_dims(tf.range(minPockets), axis = 0)

class Predictor:
  def getLigandSiteModel(self, ligand_site_ckp_path):
    ligand_site_model = MaSIF_ligand_site(
      params["max_distance"],
      params["n_classes"],
      feat_mask=params["feat_mask"],
      keep_prob = 1.0,
      n_conv_layers = 1
    )
    ligand_site_model.compile(optimizer = ligand_site_model.opt,
      loss = ligand_site_model.loss_fn,
      metrics=['accuracy']
    )
    ligand_site_model.load_weights(ligand_site_ckp_path)
    return ligand_site_model
    
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
    self.n_pockets = self.mask.shape[0]
    
    self.indices = np.load(os.path.join(pdb_dir, 'p1_list_indices.npy'), encoding="latin1", allow_pickle = True)
    self.indices = pad_indices(selfindices, 200)
    
    self.data_dict = {'input_feat' : self.input_feat, 'rho_coords' : self.rho_coords,
                   'theta_coords' : self.theta_coords, 'mask' : self.mask}
    def getFlatDataFromDict(key, sample):
      data = self.data_dict[key]
      return np.take(data, sample, axis=0).flatten()
    def getDataSampleTemp(sample):
      temp_fn = lambda key : getFlatDataFromDict(key, sample)
      flat_list = list(map(temp_fn, data_order))
      return tf.expand_dims(tf.concat(flat_list, axis=0), axis=0)
    self.getDataSample = lambda sample : getDataSampleTemp(sample)
  
  # Run MaSIF_ligand_site on all points in pdb, return probablity value for each site
  def getLigandSiteProbs(self):
    ligand_site_pred_list = []
    fullSamples = self.n_pockets // minPockets

    print('{} batches to run on ligand_site'.format(fullSamples))
    before_time = process_time()

    for i in range(fullSamples):
      if i % 10 == 0:
        done = 100.0 * i/fullSamples
        print('{} of {} batches completed. {}% done...'.format(i, fullSamples, round(done)))
      sample = range(minPockets * i, minPockets * (i+1))
      temp_X = self.getDataSample(sample)
      temp_pred = tf.sigmoid(tf.squeeze(self.ligand_site_model(temp_X, gen_sample)))
      ligand_site_pred_list.append(temp_pred)

    i = fullSamples
    n_leftover = self.n_pockets % minPockets
    valid = tf.range(minPockets * i, minPockets * i + n_leftover)
    #garbage = tf.zeros([minPockets - n_leftover], dtype=tf.int32)
    #sample = tf.expand_dims(tf.concat([valid, garbage], axis=0), axis=0)

    garbage = tf.range(minPockets * (i-1) + n_leftover, minPockets * i)
    sample = tf.expand_dims(tf.concat([garbage, valid], axis=0), axis=0)
    
    temp_X = self.getDataSample(sample)
    temp_pred = tf.sigmoid(tf.squeeze(self.ligand_site_model(temp_X, gen_sample)))
    ligand_site_pred_list.append(temp_pred[-n_leftover:])

    after_time = process_time()
    print('100% of batches completed in {} seconds.'.format(round(after_time - before_time)))

    return tf.concat(ligand_site_pred_list, axis = 0)
  '''
  def getLigandSiteProbs(self):
    ligand_site_pred_list = []
    fullSamples = self.n_pockets // minPockets
    
    ####
    randIdx = np.arange(self.n_pockets)
    np.random.shuffle(randIdx)
    order = np.argsort(randIdx)
    ####
    
    print('{} batches to run on ligand_site'.format(fullSamples))
    before_time = process_time()

    for i in range(fullSamples):
      if i % 10 == 0:
        done = 100.0 * i/fullSamples
        print('{} of {} batches completed. {}% done...'.format(i, fullSamples, round(done)))
      
      ####
      sample = randIdx[minPockets * i : minPockets * (i+1)]
      ####
      
      temp_X = self.getDataSample(sample)
      temp_pred = tf.sigmoid(tf.squeeze(self.ligand_site_model(temp_X, gen_sample)))
      ligand_site_pred_list.append(temp_pred)

    i = fullSamples
    n_leftover = self.n_pockets % minPockets
    valid = tf.range(minPockets * i, minPockets * i + n_leftover)
    garbage = tf.zeros([minPockets - n_leftover], dtype=tf.int32)
    sample = tf.expand_dims(tf.concat([valid, garbage], axis=0), axis=0)

    temp_X = self.getDataSample(sample)
    temp_pred = tf.sigmoid(tf.squeeze(self.ligand_site_model(temp_X, gen_sample)))
    ligand_site_pred_list.append(temp_pred[:n_leftover])

    after_time = process_time()
    print('100% of batches completed in {} seconds.'.format(round(after_time - before_time)))

    ligand_site_probs = tf.concat(ligand_site_pred_list, axis = 0)
    ####
    return tf.gather(ligand_site_probs, order)
    ####
  '''
  # Wrapper function for 
  def predictPocketPoints(self, threshold = None):
    ligand_site_probs = self.getLigandSiteProbs()
    
    if threshold is None:
      threshold = self.threshold
    pocket_points = tf.where(ligand_site_probs > threshold)
    return tf.squeeze(pocket_points)
  
  # Get geometric coordinates of PDB
  def getXYZCoords(pdb_dir):
    X = np.load(os.path.join(pdb_dir, "p1_X.npy"))
    Y = np.load(os.path.join(pdb_dir, "p1_Y.npy"))
    Z = np.load(os.path.join(pdb_dir, "p1_Z.npy"))
    xyz_coords = np.vstack([X, Y, Z]).T
    return xyz_coords
  
  # Get input to MaSIF_ligand, using pocket_points
  def getLigandX(self, pocket_points):
    getDataFromDict = lambda key : tf.reshape(tf.gather(self.data_dict[key], pocket_points, axis = 0), [-1])
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
      temp_prob = tf.squeeze(self.ligand_model(X))
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
    
    #ligand_site_X = self.getLigandSiteX()
    #pocket_points = self.predictPocketPoints(ligand_site_X)
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
