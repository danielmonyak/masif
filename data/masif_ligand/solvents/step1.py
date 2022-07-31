import numpy as np
import os

dir = 'solvent_pdbs'
files = os.listdir(dir)
outdir = 'newLists'
if not os.path.exists(outdir):
    os.mkdir(outdir)

for fi in files:
    arr = np.loadtxt(os.path.join(dir, fi), dtype=str, delimiter=',')
    np.savetxt(os.path.join(outdir, fi), arr, fmt='%s')
