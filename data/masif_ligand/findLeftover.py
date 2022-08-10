import os
import numpy as np

all_pdbs = np.loadtxt('newPDBs/filtered_pdbs.txt', dtype=str)
done = os.listdir('/data02/daniel/masif/masif_ligand/data_preparation/04a-precomputation_12A/precomputation/')

todo = all_pdbs[~np.isin(all_pdbs, done)]
np.savetxt('todo.txt', todo, fmt='%s')
