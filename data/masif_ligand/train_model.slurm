#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --time=48:0:0
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --mem 16384

#source /work/upcorreia/bin/load_masif_environment_gpu.sh

source /apps01/anaconda3/etc/profile.d/conda.sh
conda activate venv_tf1-15


#export MONET_DIR=/home/gainza/lpdi_fs/seednet/monet_seeder/
#export PYTHONPATH=$PYTHONPATH:$MONET_DIR:./

masif_root=$(git rev-parse --show-toplevel)
masif_source=$masif_root/source
export PYTHONPATH=$PYTHONPATH:$masif_source

#srun python -u $masif_source/masif_ligand/masif_ligand_train.py
python -u $masif_source/masif_ligand/masif_ligand_train_new.py
