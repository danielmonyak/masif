source /apps01/anaconda3/etc/profile.d/conda.sh
conda activate venv_tf1-15

masif_root=$(git rev-parse --show-toplevel)
masif_source=$masif_root/source/
masif_data=$masif_root/data/
export PYTHONPATH=$PYTHONPATH:$masif_source:$masif_data/masif_site/

python -u $masif_source/masif_site/masif_site_train.py $1 > train_nn.out 2>train_nn.err &
disown -h $!
echo $! > train_nn_pid.txt
