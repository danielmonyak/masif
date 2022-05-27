# Record of Masif Installation and Use

## Software Management/Installation:
Two virtual environments are required, as StrBioInfo needs to run on Python 2.7, and is only used for two scripts, but everything else needs to run on Python 3. The Python 3 environment was specifically set up to support tensorflow, as well. Both environments were created with anaconda:
conda create -n venv_latest tensorflow
conda create -n venv_sbi python=2.7

### venv_latest
The following anaconda install commands were used to install third party dependencies:
```
conda activate venv_latest
conda install -c schrodinger pdb2pqr
conda install -c bioconda msms
conda install -c conda-forge scikit-learn
conda install -c conda-forge ipython
conda install -c conda-forge networkx
```
#### PyMesh

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
