pdb=$1

tf_env=venv_tf_new
env_path=$HOME/miniconda3/envs/$tf_env

# Set environmental variables
export APBS_BIN=$HOME/software/APBS-3.4.1.Linux/bin/apbs
export MULTIVALUE_BIN=$HOME/software/APBS-3.4.1.Linux/share/apbs/tools/bin/multivalue
export PDB2PQR_BIN=$env_path/bin/pdb2pqr
export PATH=$HOME/software/reduce/build/reduce_src:$PATH
export REDUCE_HET_DICT=$HOME/software/reduce/reduce_wwPDB_het_dict.txt
export PYMESH_PATH=$HOME/software/PyMesh
export MSMS_BIN=$env_path/bin/msms
export PDB2XYZRN=$env_path/bin/pdb_to_xyzrn

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $tf_env

masif_root=$(git rev-parse --show-toplevel)
masif_source=$masif_root/source
export PYTHONPATH=$PYTHONPATH:$masif_source
PDB_ID=$(echo $pdb| cut -d"_" -f1)
CHAIN1=$(echo $pdb| cut -d"_" -f2)
CHAIN2=$(echo $pdb| cut -d"_" -f3)

python $masif_source/data_preparation/00-pdb_download.py $pdb

source /apps01/anaconda3/etc/profile.d/conda.sh
conda activate venv_sbi

python2.7 $masif_source/data_preparation/00b-generate_assembly.py $pdb
python2.7 $masif_source/data_preparation/00c-save_ligand_coords.py $pdb ligand all_ligands

conda deactivate

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $tf_env

# Need to add "</dev/null" because this step may fail, and activate an interactivate python shell, which will cause the scheduler to fail
python $masif_source/data_preparation/01-pdb_extract_and_triangulate.py $PDB_ID\_$CHAIN1 masif_ligand </dev/null
python $masif_source/data_preparation/04-masif_precompute.py masif_ligand $pdb

conda deactivate

echo Finished
