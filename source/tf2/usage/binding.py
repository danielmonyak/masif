import os
import numpy as np
from sklearn.cluster import KMeans
from scipy import spatial
from default_config.util import *
from tf2.usage.predictor import Predictor
'''
precom_dir = '/data02/daniel/masif/data_preparation/04a-precomputation_12A/precomputation'
ligand_model_path = '/home/daniel.monyak/software/masif/source/tf2/masif_ligand/kerasModel/savedModel'
ligand_site_ckp_path = '/home/daniel.monyak/software/masif/source/tf2/ligand_site/kerasModel/ckp'
pred = Predictor(ligand_model_path, ligand_site_ckp_path)
'''
# Calculate the Within-Cluster-Sum of Squared Errors (WSS) for different values of k
def calculate_WSS(points, kmax):
  sse = []
  for k in range(1, kmax+1):
    kmeans = KMeans(n_clusters = k).fit(points)
    centroids = kmeans.cluster_centers_
    pred_clusters = kmeans.predict(points)
    curr_sse = 0
    
    # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
    for i in range(len(points)):
      curr_center = centroids[pred_clusters[i]]
      curr_sse += (points[i, 0] - curr_center[0]) ** 2 + (points[i, 1] - curr_center[1]) ** 2
      
    sse.append(curr_sse)
  return sse

# Find "elbow" of plot by finding max of second derivative (greatest inflection)
# Returns best "k" for k-means clustering
def findBestK(coord_list, kmax=10):
  sse = calculate_WSS(coord_list, kmax)
  getDifs = lambda x : [x[i]-x[i+1] for i in range(len(x)-1)]
  sse_dif_difs = np.array(getDifs(getDifs(sse)))
  best_k = np.argmax(sse_dif_difs) + 2
  return best_k
'''
# Run predictor on pdb, find best k, cluster coordinates
def predictRaw(pdb):
  pdb_dir = os.path.join(precom_dir, pdb)
  ligand_pred, coord_list = pred.predict(pdb_dir)
  
  best_k = findBestK(coord_list)
  kmeans = KMeans(n_clusters = best_k).fit(coord_list)
  binding_loc = kmeans.cluster_centers_

  return (ligand_pred, binding_loc)

def predict(pdb = '1C75_A_'):
  ligand_pred, binding_loc = predictRaw(pdb)
  print('{} binds {} at \n{}'.format(pdb.split('_')[0], ligand_pred, binding_loc))
'''
