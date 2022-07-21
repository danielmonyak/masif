import os
import numpy as np
import sys

from SBI.structure import PDB
from default_config.masif_opts import masif_opts
params = masif_opts['ligand']

# Ligands of interest
ligands = masif_opts['ligand_list']
extra_ligands = masif_opts['extra_ligands']

files = os.listdir(params["assembly_dir"])

ret_list = []

for i, fi in enumerate(files):
    if i % 100 == 0:
        print(i)
    try:
        structure = PDB(
            os.path.join(params["assembly_dir"], fi)
        )
    except:
        continue
    for chain in structure.chains:
        for het in chain.heteroatoms:
            if het.type in ligands or het.type in extra_ligands:
                ret_list.append(het.type)

np.save('ret_list.npy', ret_list)

