source /apps01/anaconda3/etc/profile.d/conda.sh
conda activate venv_tf

masif_root=$(git rev-parse --show-toplevel)
masif_source=$masif_root/source
export PYTHONPATH=$PYTHONPATH:$masif_source


sed -i '/[A-Z]/d' results/recall.txt
sed -i '/[A-Z]/d' results/precision.txt
sed -i '/[A-Z]/d' results/lig_true.txt
sed -i '/[A-Z]/d' results/lig_pred.txt


python analysis.py
