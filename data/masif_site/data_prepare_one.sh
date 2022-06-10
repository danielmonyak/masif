#!/bin/bash

tf_env=venv_latest
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
conda activate $tf_env



masif_root=$(git rev-parse --show-toplevel)
masif_source=$masif_root/source/
masif_matlab=$masif_root/source/matlab_libs/
export PYTHONPATH=$PYTHONPATH:$masif_source
if [ "$1" == "--file" ]
then
	echo "Running masif site on $2"
	PPI_PAIR_ID=$3
	PDB_ID=$(echo $PPI_PAIR_ID| cut -d"_" -f1)
	CHAIN1=$(echo $PPI_PAIR_ID| cut -d"_" -f2)
	CHAIN2=$(echo $PPI_PAIR_ID| cut -d"_" -f3)
	FILENAME=$2
	mkdir -p data_preparation/00-raw_pdbs/
	cp $FILENAME data_preparation/00-raw_pdbs/$PDB_ID\.pdb
else
	PPI_PAIR_ID=$1
	PDB_ID=$(echo $PPI_PAIR_ID| cut -d"_" -f1)
	CHAIN1=$(echo $PPI_PAIR_ID| cut -d"_" -f2)
	CHAIN2=$(echo $PPI_PAIR_ID| cut -d"_" -f3)
	python -W ignore $masif_source/data_preparation/00-pdb_download.py $PPI_PAIR_ID
fi

if [ -z $CHAIN2 ]
then
    echo "Empty"
    python -W ignore $masif_source/data_preparation/01-pdb_extract_and_triangulate.py $PDB_ID\_$CHAIN1
else
    python -W ignore $masif_source/data_preparation/01-pdb_extract_and_triangulate.py $PDB_ID\_$CHAIN1
    python -W ignore $masif_source/data_preparation/01-pdb_extract_and_triangulate.py $PDB_ID\_$CHAIN2
fi
python $masif_source/data_preparation/04-masif_precompute.py masif_site $PPI_PAIR_ID

conda deactivate
echo Finished
