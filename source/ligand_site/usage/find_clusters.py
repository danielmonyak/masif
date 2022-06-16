from predictor import *
from sklearn.cluster import KMeans

ligand_model_path = '/home/daniel.monyak/software/masif/source/tf2/kerasModel/savedModel'
#ligand_site_model_path = '/software/masif/source/ligand_site/kerasModel/savedModel'
ligand_site_ckp_path = '/home/daniel.monyak/software/masif/source/ligand_site/kerasModel/ckp'

pdb = '1C75_A_'
precom_dir = '/data02/daniel/masif/data_preparation/04a-precomputation_12A/precomputation'
pred = Predictor(ligand_model_path, ligand_site_ckp_path)

pdb_dir = os.path.join(precom_dir, pdb)

ligand_pred, coord_list = pred.predict(pdb_dir)

#kmeans = KMeans().fit(coord_list)

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

kmax = 10
sse = calculate_WSS(coord_list, kmax)
#import matplotlib.pyplot as plt
#plt.plot(range(1, kmax+1), sse)
#plt.savefig('sse_plot.png')

getDifs = lambda x : [x[i]-x[i+1] for i in range(len(x)-1)]
sse_difs = getDifs(sse)
sse_dif_difs = np.array(getDifs(sse_difs))

best_k = np.argmax(sse_dif_difs) + 2
