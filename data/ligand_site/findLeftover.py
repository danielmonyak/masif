import os
import numpy as np

#all_pdbs = np.loadtxt('../masif_ligand/newPDBs/filtered_pdbs.txt', dtype=str)
all_pdbs = np.loadtxt('../masif_ligand/newPDBs/using_pdbs_final_reg.txt', dtype=str)
done = os.listdir('/data02/daniel/masif/masif_ligand/data_preparation/04a-precomputation_9A/precomputation/')

todo = all_pdbs[~np.isin(all_pdbs, done)]
np.savetxt('todo.txt', todo, fmt='%s')
