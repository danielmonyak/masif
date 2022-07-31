source /apps01/anaconda3/etc/profile.d/conda.sh
conda activate venv_tf1-15

masif_root=$(git rev-parse --show-toplevel)
masif_source=$masif_root/source
export PYTHONPATH=$PYTHONPATH:$masif_source

python -u filter.py > filter.out 2>&1 &
