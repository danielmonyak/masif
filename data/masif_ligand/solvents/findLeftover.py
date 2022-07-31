import os
import numpy as np

all = np.loadtxt('solvent_PDBs_UNIQUE.txt', dtype=str)
done = os.listdir('/data02/daniel/masif/masif_ligand/data_preparation/00c-ligand_coords')

#all = np.char.partition(all, '_')[:,0]
done = np.char.partition(done, '_')[:,0]

todo = all[~np.isin(np.char.partition(all, '_')[:,0], done)]
np.savetxt('todo.txt', todo, fmt='%s')
