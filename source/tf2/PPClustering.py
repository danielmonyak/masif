import os
import numpy as np
from sklearn.cluster import KMeans
import default_config.util as util
from default_config.masif_opts import masif_opts

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

def getPPClusters(pocket_points, xyz_coords):
  pocket_points_coords = xyz_coords[pocket_points]
  best_k = findBestK(pocket_points_coords)
  cluster_labels = KMeans(n_clusters = best_k).fit_predict(pocket_points_coords)

  pp_list = []
  for label in range(best_k):
      pp_temp = pocket_points_pred[cluster_labels == label]
      if len(pp_temp) >= masif_opts['ligand']['minPockets']:
          pp_list.append(pp_temp)
  return pp_list
