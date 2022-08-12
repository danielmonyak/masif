source $miniconda_activate
conda activate venv_tf_new

masif_root=$(git rev-parse --show-toplevel)
masif_source=$masif_root/source
export PYTHONPATH=$PYTHONPATH:$masif_source

job_name=train_model

script=${job_name}.py

#####
if [[ -f train_vars.pickle ]]; then rm train_vars.pickle; fi
python $masif_source/tf2/input_helper_epochs.py
#####

python -u $script > ${job_name}.out 2>${job_name}.err &
disown -h $!
echo $! > ${job_name}_pid.txt
