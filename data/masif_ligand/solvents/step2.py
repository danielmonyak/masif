import numpy as np
import os

dir = 'newLists'
files = os.listdir(dir)

for fi in files:
    arr = np.loadtxt(os.path.join(dir, fi), dtype=str)
    arr = arr[np.char.partition(arr, '_')[:,-1] == '1']
    arr = np.char.replace(arr, '_1', '_A_')
    np.savetxt(fi, arr, fmt='%s')

