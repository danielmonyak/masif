import os
import numpy as np
import sys

from SBI.structure import PDB
from default_config.masif_opts import masif_opts

params = masif_opts["ligand"]
all_ligands = masif_opts['all_ligands']
reg_ligands = masif_opts['ligand_site']
solvents = masif_opts['solvents']

all_lig_count = 0
reg_lig_count = 0
solvents_count = 0

bad_pdbs = []

#pdb_list = np.loadtxt('filtered_pdbs.txt', dtype=str)
pdb_files = os.listdir('/data02/daniel/masif/masif_ligand/data_preparation/00b-pdbs_assembly')

n_pdbs = len(pdb_files)
for k, fi in enumerate(pdb_files):
    if k == 10:
        break
    if k % 500 == 0:
        print('Working on {} of {} proteins...'.format(k, n_pdbs))
    try:
        structure = PDB(
            os.path.join(params["assembly_dir"], fi)
        )
    except:
        bad_pdbs.append(fi)
        continue
    for chain in structure.chains:
        for het in chain.heteroatoms:
            if het.type in all_ligands:
                all_lig_count += 1
                if het.type in reg_ligands:
                    reg_lig_count += 1
                if het.type in solvents:
                    solvents_count += 1

print('all_lig_count:', all_lig_count)
print('reg_lig_count:', reg_lig_count)
print('solvents_count:', solvents_count)
