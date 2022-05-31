# Record of Masif Installation and Use

## Software Management/Installation:
Two virtual environments are required, as StrBioInfo needs to run on Python 2.7, and is only used for two scripts, but everything else needs to run on Python 3. The Python 3 environment was specifically set up to support tensorflow, as well. Both environments were created with anaconda:
conda create -n venv_latest tensorflow
conda create -n venv_sbi python=2.7

### venv_latest
The following anaconda install commands were used to install some of the third party dependencies:
```
conda activate venv_latest
conda install -c schrodinger pdb2pqr
conda install -c bioconda msms
conda install -c conda-forge scikit-learn
conda install -c conda-forge ipython
conda install -c conda-forge networkx
```
Make sure to install Tensorflow-GPU if you intend to use a GPU, as recommended by the authors:
```
conda install -c anaconda tensorflow-gpu -n venv_latest
```

#### Reduce
Clone from repository and follow build instructions while venv_latest is activated:
```
git clone https://github.com/rlabduke/reduce.git
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

#### APBS
Download pre-built binaries: https://github.com/Electrostatics/apbs/releases <br>
Look under **Assets**

### venv_sbi
StrBioInfo must be installed with Pip, without installing the dependencies at the same time:
```
conda activate venv_sbi
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

### Data Preparation

Alter the "source/default_config/masif_opts.py" file to change the directory where the output files are generated. Up to 400 GB of disk space will be needed. In my version of the script, I have added a line at the top where you can change the value of "$basedir", which will alter the output directory just for masif_ligand. <br>

data_prepare_one.sh had to be run for each protein, so manual scheduling with a script was used, as Slurm is not available. Jobs were run such that a maximum of 8 were running at any one time.
```
./run_data_prepare.sh
```
Runs the scheduling script in the background and stores the pid of the job in "schedule_data_prepare_pid.txt". Examine the current jobs with "ps aux". Proteins that have been started and then finished are listed in "started_proteins.txt" and "finished_proteins.txt". Be aware that there are a few duplicates in "sequence_split_list.txt", so there will be less output folders than proteins in the list.

#### Make TF Records

```
./run_make_tfrecord.sh
```
Runs the make_tfrecord.slurm script in the background (not as a slurm job) and stores the pid in "make_tfrecord_pid.txt". It splits the data into training, test, and validation sets, generating output in "$basedir/data_preparation/tfrecords".

### Training Model
```
./run_train_model.sh
```
Runs the train_model.slurm script in the backround and stores the pid in "train_model_pid.txt". It will automatically use the TF Records that were generated earlier.
