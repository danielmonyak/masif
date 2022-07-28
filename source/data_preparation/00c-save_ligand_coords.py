import os
import numpy as np
import sys

from SBI.structure import PDB
from default_config.masif_opts import masif_opts

if len(sys.argv) > 2:
    masif_app = sys.argv[2]
else:
    masif_app = 'ligand'

params = masif_opts[masif_app]

in_fields = sys.argv[1].split("_")
pdb_id = in_fields[0]

# Edited by Daniel Monyak
# Added try-except blocks in the "makedir" if statements so that there aren't multi-processing bugs

if not os.path.exists(params["ligand_coords_dir"]):
    try:
        os.mkdir(params["ligand_coords_dir"])
    except:
        pass

# Ligands of interest

if len(sys.argv) > 3:
    lig_l = sys.argv[3]
else:
    lig_l = 'ligand_list'

ligands = masif_opts[lig_l]    

structure_ligands_type = []
structure_ligands_coords = []
try:
    structure = PDB(
        os.path.join(params["assembly_dir"], "{}.pdb".format(pdb_id))
    )
except:
    print("Problem with opening structure", pdb)
for chain in structure.chains:
    for het in chain.heteroatoms:
        # Check all ligands in structure and save coordinates if they are of interest
        if het.type in ligands:
            structure_ligands_type.append(het.type)
            structure_ligands_coords.append(het.all_coordinates)

if len(structure_ligands_type) > 0:
    np.save(
        os.path.join(
            params["ligand_coords_dir"], "{}_ligand_types.npy".format(pdb_id)
        ),
        structure_ligands_type
    )
    np.save(
        os.path.join(
            params["ligand_coords_dir"], "{}_ligand_coords.npy".format(pdb_id)
        ),
        structure_ligands_coords
    )
