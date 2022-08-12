ource $miniconda_activate
conda activate venv_tf_new

masif_root=$(git rev-parse --show-toplevel)
masif_source=$masif_root/source
export PYTHONPATH=$PYTHONPATH:$masif_source

python -u evaluate.py $1 $2 > $1/${2}_evaluate.out 2>evaluate.err &
disown -h $!
echo $! > evaluate_pid.txt
