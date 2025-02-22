#!/bin/bash

tf_env=venv_tf
env_path=/apps01/anaconda3/envs/$tf_env

export APBS_BIN=$HOME/software/APBS-3.4.1.Linux/bin/apbs
export MULTIVALUE_BIN=$HOME/software/APBS-3.4.1.Linux/share/apbs/tools/bin/multivalue
export PDB2PQR_BIN=$env_path/bin/pdb2pqr
export PATH=$HOME/software/reduce/reduce_src:$PATH
export REDUCE_HET_DICT=$HOME/software/reduce/reduce_wwPDB_het_dict.txt
export PYMESH_PATH=$HOME/software/PyMesh
export MSMS_BIN=$env_path/bin/msms
export PDB2XYZRN=$env_path/bin/pdb_to_xyzrn

source /apps01/anaconda3/etc/profile.d/conda.sh

masif_root=$(git rev-parse --show-toplevel)
masif_source=$masif_root/source/
masif_matlab=$masif_root/source/matlab_libs/
export PYTHONPATH=$PYTHONPATH:$masif_source

conda activate venv_tf
python $masif_source/data_preparation/00-pdb_download.py $1
conda deactivate venv_tf

conda activate venv_sbi
python2.7 $masif_source/data_preparation/00b-generate_assembly.py $1 site
python2.7 $masif_source/data_preparation/00c-save_ligand_coords.py $1 site
conda deactivate

echo Finished
