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

data_prepare_one.sh had to be run for each protein, so manual scheduling with a script was used, as Slurm is not available. Jobs were run such that a maximum of 8 were running at any one time.
