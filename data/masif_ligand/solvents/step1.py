import numpy as np
import os
files = os.listdir()

for fi in files:
    arr = np.loadtxt(fi, dtype=str, delimiter=',')
    arr = arr[np.char.partition(arr, '_')[:,-1] == '1']
    arr = np.char.replace(arr, '_1', '_A_')
    np.savetxt(fi, arr, fmt='%s')

