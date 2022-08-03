import numpy as np
import os
from default_config.masif_opts import masif_opts

params = masif_opts['ligand']

all_pdbs = np.loadtxt('filtered_pdbs.txt', dtype=str)
listdir = 'lists'
if not os.path.exists(listdir):
    os.mkdir(listdir)

np.random.shuffle(all_pdbs)
train = int(len(all_pdbs) * params["train_fract"])
val = int(len(all_pdbs) * params["val_fract"])
test = int(len(all_pdbs) * params["test_fract"])
print("Train", train)
print("Validation", val)
print("Test", test)
train_pdbs = all_pdbs[:train]
val_pdbs = all_pdbs[train : train + val]
test_pdbs = all_pdbs[train + val : train + val + test]

np.save(os.path.join(listdir, 'train_pdbs.npy'), train_pdbs)
np.save(os.path.join(listdir, 'val_pdbs.npy'), val_pdbs)
np.save(os.path.join(listdir, 'test_pdbs.npy'), test_pdbs)
