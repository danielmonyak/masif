import os
import numpy as np
import sys

from SBI.structure import PDB
from default_config.masif_opts import masif_opts

in_fields = sys.argv[1].split("_")
pdb_id = in_fields[0]

# Edited by Daniel Monyak
# Added try-except blocks in the "makedir" if statements so that there aren't multi-processing bugs

if not os.path.exists(masif_opts["ligand"]["ligand_coords_dir"]):
    try:
        os.mkdir(masif_opts["ligand"]["ligand_coords_dir"])
    except:
        pass

# Ligands of interest
ligands = ["ADP", "COA", "FAD", "HEM", "NAD", "NAP", "SAM"]

######## Edited by Daniel Monyak
#structure_ligands_type = []
#structure_ligands_coords = []
ligand_dict = {}

try:
    structure = PDB(
        os.path.join(masif_opts["ligand"]["assembly_dir"], "{}.pdb".format(pdb_id))
    )
except:
    print("Problem with opening structure", pdb)
for chain in structure.chains:
    for het in chain.heteroatoms:
        # Check all ligands in structure and save coordinates if they are of interest
        if het.type in ligands:
            # Edited by Daniel Monyak
            if het.type in ligand_dict:
                ligand_dict[het.type].append(het.all_coordinates)
            else:
                ligand_dict[het.type] = [het.all_coordinates]
                #structure_ligands_type.append(het.type)
                #structure_ligands_coords.append(het.all_coordinates)

structure_ligands_type = list(ligand_dict.keys())
structure_ligands_coords = list(map(lambda ligand_type : np.concatenate(ligand_dict[ligand_type], axis = 0), structure_ligands_type))                
########

np.save(
    os.path.join(
        masif_opts["ligand"]["ligand_coords_dir"], "{}_ligand_types.npy".format(pdb_id)
    ),
    structure_ligands_type,
)
np.save(
    os.path.join(
        masif_opts["ligand"]["ligand_coords_dir"], "{}_ligand_coords.npy".format(pdb_id)
    ),
    structure_ligands_coords,
)
