source /apps01/anaconda3/etc/profile.d/conda.sh
conda activate venv_tf1-15

masif_root=$(git rev-parse --show-toplevel)
masif_source=$masif_root/source
export PYTHONPATH=$PYTHONPATH:$masif_source

job_name=interpret_results

script=${job_name}_prob.py 

python -u $script > ${job_name}_prob.out 2>&1
#python -i $script
