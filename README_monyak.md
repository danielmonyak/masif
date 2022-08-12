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
Install third party dependencies:
```
conda install -c conda-forge biopython
conda install -c schrodinger pdb2pqr
conda install -c bioconda msms
conda install -c conda-forge scikit-learn
conda install -c conda-forge ipython
conda install -c conda-forge networkx
```

#### PyMesh
Build from source: https://github.com/PyMesh/PyMesh, do not install with anaconda <br>
Don't clone each third party repository separately, use their instructions while **venv_tf_new** is activated:
```
git clone https://github.com/PyMesh/PyMesh.git
cd PyMesh
git submodule update --init
```
```
cd third_party
python build.py all
```
```
cd ..
mkdir build
cd build
cmake ..
```
```
make
make tests
```
```
python setup.py install
```

#### Reduce
Clone repository: https://github.com/rlabduke/reduce <br>
Follow build instructions while **venv_tf_new** is activated:
```
git clone https://github.com/rlabduke/reduce.git
```
Follow build instructions in Reduce README
```
cd reduce
mkdir build
cd build
cmake ..
```
```
make
```

#### Openbabel
While **venv_tf_new** is activated:
```
conda install -c openbabel openbabel
```

#### Important Note
When using Tensorflow, if it is giving error messages about not being able to convert a function to Autograph because of the "gast" package, then run this and disregard the messages about missing dependencies:
```
pip install gast==0.3.3
```

### APBS
Does not need to be installed in any environment, just need to download pre-built binaries for APBS 3.4.1: https://github.com/Electrostatics/apbs/releases <br>
Look under **Assets**: APBS-3.4.1.Linux.zip
<br><br><br>

You may have to edit the paths of the enviornmental variables in "data/masif_ligand/data_prepare_one.sh" based you install everything.

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

### Important Notes/Terms
To see the current running jobs:
```
ps -u $username -o pid,%mem,%cpu,user,cmd
```
masif_opts - dictionary in source/default_config/masif_opts.py with useful paths and constants, which is imported in all Python scripts <br>

TF1 - Tensorflow 1 - not important, do not use these <br>
TF2 - Tensorflow 2 (also involving the use of Keras) <br>

The original models for MaSIF were trained using TF1, but I transitioned everything over to TF2/Keras. <br>

LOI - Ligands of interest - ADP, COA, FAD, HEM, NAD, NAP, SAM <br>
Solvents - Solvent molecules often in crystallization process - IMP, EPE, FMN, TRS, PGE, ACT, NAG, EDO, GOL, SO4, PO4 <br>

n_samples - number of patches

See masif_opts for the LOI and solvent lists <br>

None of the currently trained models were trained for dealing with solvents or protein sites with solvents, but about 3000 extra PDBs were downloaded and prepared, and exist in masif_opts["ligand"]["masif_precomputation_dir"] <br>

#### Scripts
Almost all scripts (basename.py) are written in Python but many of them should be run using the corresponding shell script (basename.sh) that activates the right virtual environment and runs the Python script in the background, and is usally in the same directory. The shell script redirects the error and output streams to files with the same base name as the script (basename.err and basename.out) in the same directory, but occasionally the error stream is just fed into the output file as well. It also creates a file that contains the PID of the Python job that is running in the background (basename_pid.txt), so that it can be terminated.<br><br>
EXAMPLE: All the of the scripts to train a model are "train_model.py" but should be run by doing:
```
./train_model.sh
```
This will create "train_model.out" and "train_model.err", as well as "train_model_pid.txt". If you need to terminate the Python process, run:
```
kill $(cat train_model_pid.txt)
```

It is a good idea to add this to your ~/.bashrc file:
```
masif_source=~/software/masif/source
export PYTHONPATH=$PYTHONPATH:$masif_source
```
This way, you can access modules in the MaSIF repo using import statements.

### Directories

**data**: Scripts for data prepartion<br>
**data/masif_ligand**: Scripts to do data preparation for MaSIF-Ligand<br>
- data_prepare_one.sh - Run all data preparation steps for a single protein structure <br>
- schedule_wait.sh, run_schedule_wait.sh, schedule_no_wait.sh, run_schedule_no_wait.sh - see section below on "Data Preparation" <br>
- findLeftover.py - generate list of PDBs that still need to run through "data_prepare_one.sh" <br>
- examineLigands.py - not important, used to examine difference in necessary distance generate pocket points, in LOI vs solvents <br><br>
**data/masif_ligand/TF1**: Scripts used for TF1 Work - not important <br>
- nn_models - TF1 saved model weights <br>
- train_model.sh - shell script to train TF1 model <br>
- old_slurm_scripts - scripts to use if Slurm is available - have not tried these <br>
- make_tfrecord.sh - shell script to make the TFRecordDataset objects that were used in TF1 training <br>
- lists/sequence_split_list.txt is old list of PDBS used for MaSIF-Ligand <br>
- lists - all contains old training, validation, and testing PDB lists used for TF1 <br>
- evaluate_test.sh - calls python script to generate preditions on the test dataset using the TF1 model <br> 
- analysis - contains scripts for analyzing results of TF1 model <br>
**data/masif_ligand/newPDBs**: **IMPORTANT** - Work that involved collection of new protein sctructures that resulted in the current, expanded train, val, and test sets <br>
- CSVs - csv files downloaded from the PDB, all structures that bind to the LOI <br>
- CSVs - csv files downloaded from the PDB, structures with solvents <br>
- makeDF.py - processes the csv files and combines them into a usable dataframe, that will be in the same directory, called df.csv, and creates all_pdbs.txt (unimportant), which contains all the structures listed in all of the csv files, but is not a filtered list <br>
- filter.py - performs filtering steps on all protein structures <br>
- using_pdbs_final_reg.txt - IMPORTANT - the new list of PDBs that bind LOIs <br>
- filtered_pdbs.txt - all structures in using_pdbs_final_reg.txt, plus PDBs that contain solvents that were filtered and downloaded <br>
- shared_old.txt - all structures that were shared between the original list of PDBs contained in data/masif_ligand/TF1/lists/sequence_split_list.txt and all of the PDBs collected and contained in data/masif_ligand/newPDBs/CSVs <br>
- examineLigands.py - not important, used to examine how many PDbs in the the final list have a LOI vs. solvent <br>
**data/ligand_site**: Scripts to do precomputation with 9A radius (for LSResNet and ligand-site) <br>
- re_precompute.sh - Run just the precomputation step with 9A - creates new directory "04a-precomputation_9A" <br>
- same scheduling scripts as in data/masif_ligand
- findLeftover.py - (see above) <br><br>

**source**: Most python source files<br>
**source/default_config**:<br>
- masif_opts.py - Contains a dictionary with useful paths and constants, which is imported in all Python scripts - Edited by me<br>
- util.py       - Contains several useful constants, functions, and classes - Created by me<br>
**source/data_preparation**: Scripts to do all preprocessing of proteins<br>
- 00-pdb_download.py - Retrieve raw PDB file<br>
- 00b-generate_assembly.py - Reads and builds the biological assembly of a protein structure<br>
- 00c-save_ligand_coords.py - Saves a protein's ligand types and coordinates in 3-d space<br>
- 01-pdb_extract_and_triangulate.py - Extract helices, protonate and charge PDB<br>
- 01b-helix_extract_and_triangulate.py - Not used in MaSIF-Ligand (preprocessing step for MaSIF-Site)<br>
- 04-masif_precompute.py - Computes features, angular coordinates of all vertices of all patches, protein coordinates in 3-d space<br>
- 04b-make_ligand_tfrecords.py - Compiles data of all proteins in TFRecordDataset objects - NOT necessary for TF2<br>
**source/masif_modules**: TF1 models and useful functions - NOT necessary for TF2<br>
**source/masif_ligand**: Scripts for training and predicting with the TF1 model<br><br>

**source/tf2**: MOST IMPORTANT - Scripts, work, and saved models for TF2 models<br>
Each of these directories contains a training file called train_model.py, with its execution scripts train_model.sh. The saved models are in directories called kerasModel in each directory.

<br><br>
**source/tf2/masif_ligand**<br>
This is a traditional approach to training a Keras model for MaSIF-Ligand. The data is imported as Numpy Arrays (then converted to ragged tensors), with each row corresponding to true binding pocket in some protein. Each row is a flattened array containing the four inputs (input_feat, rho_coords, theta_coords, mask), which are then reshaped inside the model (see MaSIF_ligand_TF2.py). The training is performed in batches of 32 pockets, the default when calling model.fit.<br>

This approach does not work when you try to include the solvent PDBs, because it is too much data to import all at once as a Numpy Array.
**source/tf2/masif_ligand/batch**<br>
This training is also in batches of 32 pockets, but is done manually. Here, the data is not imported all at once. Each training sample is in this shape: (input_feat, rho_coords, theta_coords, mask). I designed this one to see if it could work, therefore opening the possiblity of including solvent pockets, because each training sample is loaded one at a time, so that the whole dataset does not have to be loaded at once.. This is very manual, under-the-hood Keras work, with a manually written training step and a summation and then averaging of gradients for each batch. I don't know why, but I cannot get it to perform as well as the the model in source/tf2/masif_ligand. As far as I know, it is the same general idea of that model, mini-batch gradient descent. <br>
One possible reason for this underperformance is that the way the original Keras model computes the gradient is by computing the loss for all the samples in a batch at once and then calculating the gradient based on that loss. In this approach, we take the average of all the gradients in the batch. **IMPORTANT**: The reason we can't compute the loss for all samples in a batch at once is that because the number of patches in every protein (and in every pocket in every protein) is different, so there is not uniform size of input. Keras models cannot handle multiple samples at one that have different input sizes.
**source/tf2/ligand_site/batch**<br>
This is a implementation of the MaSIF-Site model, adapted to predict ligand biding sites instead of PPI sites. The input is mostly the same as MaSIF-ligand, except that the all the patches of a protein are used instead of just the pocket patches. Therefore, it designed for 9A radius precomputed data, rather than the 12A radius precomputed data used for MaSIF-ligand. Also, in addition to the four basic inputs, an additional "indices" tensor is passed. It contains the indices of every vertex in a patch, which is important in the convolutional layers. <br>
The input is in this shape: ((input_feat, rho_coords, theta_coords, mask), indices) It consists of three convolutional layers, the first of which is just like the convolutional layer in MaSIF-ligand. The input to this layer is n_samples x 100 x 5 (100 vertices for 9A radius) and the output is n_samples x 5. Then, the indices tensor is used to rebuild each patch into a tensor that is n_samples x 100 x 5, where each sample (patch) now contains those 5 artificial features (outputed after the 1st convolutional layer) for each of its vertices. <br>
This is then run through the next convolutional layer, and the same patch rebuilding takes place so that it can go through the third convolutional layer, adn then finally goes through a final hidden dense layer, and then goes to the output layer. <br>
This model gives a prediction for every single patch on whether or not it is in a pocket or not. <br>
The training is performed the same way as source/tf2/masif_ligand/batch, in that is manual aggregation of gradients and the data is loaded one at a time for each protein.
**source/tf2/LSResNet**<br>
The inputs to this model are the four basic inputs plus XYZ coordinates for the protein, in this shape ((input_feat, rho_coords, theta_coords, mask), xyz_coords). After running through the first convolutional layer (regular MaSIF angular convolutional layer), the data is in the shape n_samples x 5, which is projected into a tensor of shpae 36x36x36x5. This idea was taken from PUResNet, which performs this projection of features using a package their developer created called tfbio. I use this package to project the pocket point true values into 36x36x36x1 for a truth map, but in the model, I do it with a custom layer called MakeGrid that can function in Tensorflow as a layer, but the result is the same. From here, the 36x36x36x5 result is fed through a set of layers in the design of a PUResNet convolutional block, and the final output is a 36x36x36x1 density map. <br>
Since each protein has to be inputted in its entirety, there is too much data to make Numpy Arrays like in source/tf2/masif_ligand. Therefore, each protein has to loaded individually. In this version, model.fit is called on each protein and performs one epoch of training, so this is stochastic gradient descent, the only example of this in all of these training directories.
**source/tf2/LSResNet/batch**<br>
This is the same model as source/tf2/LSResNet, but the training is done in batches like in source/tf2/masif_ligand/batch and source/tf2/ligand_site/batch, with manual aggregation of gradients.
**source/tf2/LSResNet/batch_unet**<br>
This model was not never successful in training, but I have left it there anyways. The training is done in manual batches, as discussed above. The architecture, however is very different. The model start out like LSResNet, with the MaSIF convolutional layer, and then projection onto the 36 length 3-D grid with the MakeGrid layer. However, instead of just feeding this into one convolutional block, it is fed into the entire U-Net architecture that PUResNet uses, which is huge, with tons of Convolutional, Identity, and Up-Sampling blocks. The output is the same as 
<br><br>
**source/tf2/evaluation**<br>

### PDB Collection/Filtering
The current list of PDBs being used is "data/masif_ligand/newPDBs/using_pdbs_final_reg.txt," and "data/masif_ligand/newPDBs/filtered_pdbs.txt" is the same list plus the ~3000 PDBs that contained solvents, were filtered, and added to the list.<br>

These steps do not need to be completed again.<br><br>

Collecting PDB IDs and cluster IDs from CSV files and filter PDBs by 30% sequence identity, so that no two PDBs in final set are in the same cluster. Also, start off with the original list of PDBs that was used for MaSIF-Ligand.
```
cd data/masif_ligand/newPDBs
python makeDF.py CSVs
python filter.py reg > filter.out 2>&1 &
```
Collect solvent PDBs and add to "reg" set for final set of PDBs:
```
python makeDF.py solvent_CSVs
python filter.py solvents > filter.out 2>&1 &
```

### Data Preparation

Alter the "source/default_config/masif_opts.py" file to change the directory where the output files are generated. Up to 400 GB of disk space will be needed.
<br>

In **data/masif_ligand**:<br>
data_prepare_one.sh has to be run for each protein, so manual scheduling with one of these script should be used if slurm is not available: <br><br>
schedule_wait.sh (RECOMMENDED) - runs a batch of 10 jobs (or as many as you want) at a time, and then waits till they all finish to schedule the next batch, checking every 20 seconds <br>
schedule_no_wait.sh - runs a batch of 15 jobs (or as many as you want) every 5 minutes. <br>

Use one of them by running either:
```
./run_schedule_wait.sh
```
or
```
./run_schedule_no_wait.sh
```

It is possible that the scheduler will fail, or you may have to stop it manually if you are running schedule_no_wait.sh and too many jobs are running:
```
kill $(cat schedule_no_wait_pid.txt)
```
In this case, to start the scheduler again without repeating the process for the PDBs already done, run the "findLeftover.py" script, which generates a "todo.txt" file. Then, change the line in the scheduler shell script
```
done < filtered_pdbs.txt
```
to
```
done < todo.txt
```
