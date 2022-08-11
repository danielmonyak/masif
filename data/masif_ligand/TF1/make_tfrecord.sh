source /apps01/anaconda3/etc/profile.d/conda.sh
#conda activate venv_latest
conda activate venv_tf1-15

masif_root=$(git rev-parse --show-toplevel)
masif_source=$masif_root/source/
masif_matlab=$masif_root/source/matlab_libs/
export PYTHONPATH=$PYTHONPATH:$masif_source

python -u $masif_source/data_preparation/04b-make_ligand_tfrecords.py > make_tf.out 2>make_tf.err &
disown -h $!
echo $! > make_tfrecord_pid.txt
