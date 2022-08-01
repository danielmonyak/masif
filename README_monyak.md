# Record of Masif Installation and Use

## Software Management/Installation:
Two virtual environments are required, and should be created using anaconda/miniconda:
```
conda create -n venv_tf_new python=3.7 tensorflow-gpu=2.4.1
conda create -n venv_sbi python=2.7 
```
**venv_tf_new** was used for most of the data preparation and training the models and **venv_sbi** was used for running StrBioInfo, since it requires Python 2.7

### venv_tf_new
Activate environment:
```
conda activate venv_tf_new
```
Fix probelm with "gast" package (this is necessary for eing able to Autograph tensorflow tf.functions:
```
pip install gast==0.3.3
```
Install third party dependencies:
```
conda install -c schrodinger pdb2pqr
conda install -c bioconda msms
conda install -c conda-forge scikit-learn
conda install -c conda-forge ipython
conda install -c conda-forge networkx
```

#### PyMesh
Build from source: https://github.com/PyMesh/PyMesh, do not install with anaconda <br>
Don't clone each third party repository separately, use their instructions:
```
git clone https://github.com/PyMesh/PyMesh.git
cd PyMesh
git submodule update --init
```
Follow build instructions in PyMesh Readme, including the **Install** section.

#### Reduce
Clone from repository and follow build instructions while **venv_latest** is activated:
```
git clone https://github.com/rlabduke/reduce.git
```

#### APBS
Download pre-built binaries: https://github.com/Electrostatics/apbs/releases <br>
Look under **Assets**

### venv_sbi
Activate environment
```
conda activate venv_sbi
```
StrBioInfo must be installed with Pip, WITHOUT installing the dependencies at the same time:
```
pip install StrBioInfo --no-deps
```
Dependencies:
```
pip show StrBioInfo

Name: StrBioInfo
Version: 0.2.2
Summary: The StructuralBioInformatics Library
Home-page: UNKNOWN
Author: Jaume Bonet
Author-email: jaume.bonet@gmail.com
License: MIT
Location: /apps01/anaconda3/envs/venv_sbi/lib/python2.7/site-packages
Requires: beautifulsoup4, pynion, lxml, numpy, scipy
Required-by:
```
Install the depedencies separately:
```
pip install beautifulsoup4==4.8.0
pip install pynion==0.0.4
pip install lxml==4.4.1
pip install numpy==1.16.5
pip install scipy==1.2.1
```

## Use of Masif

TF1 - Tensorflow 1
TF2 - Tensorflow 2 (also involving the use of Keras)

### Important Directories

**data**: Scripts for data prepartion<br>
**data/masif_ligand**: Scripts to do data preparation for MaSIF-Ligand<br>

**source**: Most python source files<br>
**source/default_config**:<br>
- masif_opts.py - Contains a dictionary with useful paths and constants, which is imported in all Python scripts - Edited by me<br>
- util.py       - Contains several useful constants, functions, and classes - Created by me<br>

**source/data_preparation**:Scripts to do all preprocessing of proteins<br>
- 00-pdb_download.py - Retrieve raw PDB file<br>
- 00b-generate_assembly.py - Reads and builds the biological assembly of a protein structure<br>
- 00c-save_ligand_coords.py - Saves a protein's ligand types and coordinates in 3-d space<br>
- 01-pdb_extract_and_triangulate.py - Extract helices, protonate and charge PDB<br>
- 01b-helix_extract_and_triangulate.py - Not used in MaSIF-Ligand (preprocessing step for MaSIF-Site)<br>
- 04-masif_precompute.py - Computes features, angular coordinates of all vertices of all patches, protein coordinates in 3-d space<br>
- 04b-make_ligand_tfrecords.py - Compiles data of all proteins in TFRecordDataset objects - NOT necessary for TF2<br>

**source/masif_modules**: TF1 models and useful functions - NOT necessary for TF2<br>
**source/masif_ligand**: Scripts for training and predicting with the TF1 model<br>

**source/tf2**: MOST IMPORTANT - Scripts, work, and saved models for TF2 models


### Data Preparation

Alter the "source/default_config/masif_opts.py" file to change the directory where the output files are generated. Up to 400 GB of disk space will be needed.
<br>

data_prepare_one.sh had to be run for each protein, so manual scheduling with a script was used, as Slurm is not available. Jobs were run such that a maximum of 8 were running at any one time.
```
./run_data_prepare.sh
```
Runs the scheduling script in the background and stores the pid of the job in "schedule_data_prepare_pid.txt". Examine the current jobs with "ps aux". Proteins that have been started and then finished are listed in "started_proteins.txt" and "finished_proteins.txt". Be aware that there are a few duplicates in "sequence_split_list.txt", so there will be less output folders than proteins in the list.
