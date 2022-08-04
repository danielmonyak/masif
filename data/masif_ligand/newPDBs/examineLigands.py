import os
import numpy as np
import sys

from SBI.structure import PDB
from default_config.masif_opts import masif_opts

params = masif_opts["ligand"]
all_ligands = masif_opts['all_ligands']
reg_ligands = masif_opts['ligand_list']
solvents = masif_opts['solvents']

all_lig_count = 0
reg_lig_count = 0
solvents_count = 0

bad_pdbs = []

#pdb_list = np.loadtxt('filtered_pdbs.txt', dtype=str)
pdb_list = np.loadtxt('using_pdbs_final_reg.txt', dtype=str)

n_pdbs = len(pdb_list)

for k, pdb_id in enumerate(pdb_list):
    if k % 50 == 0:
        print('Working on {} of {} proteins...'.format(k, n_pdbs))
    try:
        all_ligand_types = np.load(
            os.path.join(
                params['ligand_coords_dir'], "{}_ligand_types.npy".format(pdb_id.split("_")[0])
            )
        ).astype(str)
    except:
        bad_pdbs.append(pdb_id)
        continue
    
    all_lig_pres = False
    reg_lig_pres = False
    solvents_pres = False
    
    for structure_ligand in all_ligand_types:
        if structure_ligand in all_ligands:
            all_lig_pres = True
            if structure_ligand in reg_ligands:
                reg_lig_pres = True
            if structure_ligand in solvents:
                solvents_pres = True
    
    if all_lig_pres:
        all_lig_count += 1
    if reg_lig_pres:
        reg_lig_count += 1
    if solvents_pres:
        solvents_count += 1

print('all_lig_count: {}'.format(all_lig_count))
print('reg_lig_count: {}'.format(reg_lig_count))
print('solvents_count: {}'.format(solvents_count))
