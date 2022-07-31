import os
import numpy as np

all = np.loadtxt('filtered_pdbs.txt', dtype=str)
done = os.listdir('/data02/daniel/masif/masif_ligand/data_preparation/04a-precomputation_12A/precomputation/')

todo = all[~np.isin(all, done)]
np.savetxt('todo.txt', todo, fmt='%s')
