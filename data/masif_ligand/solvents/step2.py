import numpy as np
import os
files = os.listdir()

pdb_list = []

for fi in files:
    arr = np.loadtxt(fi, dtype=str)
    np.savetxt(fi, arr, fmt='%s')

