source /home/daniel.monyak/miniconda3/etc/profile.d/conda.sh
conda activate venv_tf_new

masif_root=$(git rev-parse --show-toplevel)
masif_source=$masif_root/source
export PYTHONPATH=$PYTHONPATH:$masif_source

job_name=train_model

script=${job_name}.py

#####
if [[ -f train_vars.py ]]; then rm train_vars.py; fi
python $masif_source/tf2/input_helper_iters.py
#####

python -u $script > ${job_name}.out 2>${job_name}.err &
disown -h $!
echo $! > ${job_name}_pid.txt
