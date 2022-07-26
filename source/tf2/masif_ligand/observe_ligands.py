import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import sys
import numpy as np
from scipy import spatial
from random import shuffle
from default_config.util import *
from tf2.read_ligand_tfrecords import _parse_function
import tensorflow as tf

params = masif_opts["ligand"]

####
#ligands = masif_opts['ligand_list']
ligand_list = masif_opts['all_ligands']
n_classes = len(ligand_list)
####

precom_dir = params["masif_precomputation_dir"]
ligand_coord_dir = params["ligand_coords_dir"]

all_pdbs = os.listdir(params["masif_precomputation_dir"])

npoints_list = []
n_pdbs = len(all_pdbs)
for i, pdb in enumerate(all_pdbs):
  pdb = pdb.rstrip('_')
  print(f'{i+1} of {n_pdbs}')
  try:
      # Load precomputed data
      input_feat = np.load(
          os.path.join(precom_dir, pdb + "_", "p1_input_feat.npy")
      )
      rho_wrt_center = np.load(
          os.path.join(precom_dir, pdb + "_", "p1_rho_wrt_center.npy")
      )
      theta_wrt_center = np.load(
          os.path.join(precom_dir, pdb + "_", "p1_theta_wrt_center.npy")
      )
      mask = np.load(
          os.path.join(precom_dir, pdb + "_", "p1_mask.npy")
      )
      X = np.load(os.path.join(precom_dir, pdb + "_", "p1_X.npy"))
      Y = np.load(os.path.join(precom_dir, pdb + "_", "p1_Y.npy"))
      Z = np.load(os.path.join(precom_dir, pdb + "_", "p1_Z.npy"))
      all_ligand_coords = np.load(
          os.path.join(
              ligand_coord_dir, "{}_ligand_coords.npy".format(pdb.split("_")[0])
          )
      )
      all_ligand_types = np.load(
          os.path.join(
              ligand_coord_dir, "{}_ligand_types.npy".format(pdb.split("_")[0])
          )
      ).astype(str)
  except:
      continue
  
  if len(all_ligand_types) == 0:
      continue  
  
  xyz_coords = np.vstack([X, Y, Z]).T
  tree = spatial.KDTree(xyz_coords)
  pocket_labels = np.zeros(
      (xyz_coords.shape[0], len(all_ligand_types)), dtype=np.int32
  )
  # Label points on surface within 3A distance from ligand with corresponding ligand type
  for j, structure_ligand in enumerate(all_ligand_types):
      ligand_coords = all_ligand_coords[j]
      pocket_points = tree.query_ball_point(ligand_coords, 3.0)
      pocket_points = list(set([pp for p in pocket_points for pp in p]))
      npoints = len(pocket_points)
      npoints_list.append(npoints)
