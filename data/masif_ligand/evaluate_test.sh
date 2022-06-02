source /apps01/anaconda3/etc/profile.d/conda.sh
conda activate venv_tf1-15

masif_root=$(git rev-parse --show-toplevel)
masif_source=$masif_root/source
export PYTHONPATH=$PYTHONPATH:$masif_source

script=masif_ligand_evaluate_test_new.py 

python -u $masif_source/masif_ligand/$script > evaluate_test.out 2>evaluate_test.err &
disown -h $!
echo $! > evaluate_test_pid.txt
