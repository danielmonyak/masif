source /apps01/anaconda3/etc/profile.d/conda.sh
conda activate venv_tf

masif_root=$(git rev-parse --show-toplevel)
masif_source=$masif_root/source
export PYTHONPATH=$PYTHONPATH:$masif_source

job_name=train_model

script=${job_name}.py 

python -u $script $1 > ${job_name}.out 2>${job_name}.err &
disown -h $!
echo $! > ${job_name}_pid.txt
