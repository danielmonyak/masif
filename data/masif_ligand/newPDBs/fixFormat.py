import numpy as np

filtered_pdbs = np.loadtxt('filtered_pdbs.txt', dtype=str)
np.savetxt('filtered_pdbs.txt', np.char.add(filtered_pdbs, '_A_'), fmt='%s')
