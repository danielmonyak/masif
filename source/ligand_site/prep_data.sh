out=/data02/daniel/masif/datasets/ligand_site
if [ ! -d $out ]; then mkdir $out; fi

source /apps01/anaconda3/etc/profile.d/conda.sh
conda activate venv_tf

masif_root=$(git rev-parse --show-toplevel)
masif_source=$masif_root/source
export PYTHONPATH=$PYTHONPATH:$masif_source

job_name=prep_data

script=${job_name}.py 

python -u $script > ${job_name}.out 2>${job_name}.err &
disown -h $!
echo $! > ${job_name}_pid.txt
