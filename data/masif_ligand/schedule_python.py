import os
import numpy as np
from time import sleep

pdbs = np.loadtxt('newPDBs/todo.txt', dtype=str)
concur = 10
sleep_time = 150

n_pdbs = len(pdbs)
i = 0
while i < n_pdbs:
    for j in range(concur):
        temp_pdb = pdbs[i]
        print(f'Running {i} of {n_pdbs} PDBs: {temp_pdb}')
        exit = os.system(f'./data_prepare_one.sh {temp_pdb} &')
        i += 1
        if i == n_pdbs:
            break
    sleep(sleep_time)

print('Finished!')
