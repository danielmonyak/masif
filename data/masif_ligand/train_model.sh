script=masif_ligand_train_new.py 

source /apps01/anaconda3/etc/profile.d/conda.sh
conda activate venv_tf1-15

masif_root=$(git rev-parse --show-toplevel)
masif_source=$masif_root/source
export PYTHONPATH=$PYTHONPATH:$masif_source

python -u $masif_source/masif_ligand/$script > train_model.out 2>train_model.err &
disown -h $!
echo $! > train_model_pid.txt
