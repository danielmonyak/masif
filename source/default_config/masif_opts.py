import tempfile
import os

basedir = "/data02/daniel/masif/masif_ligand/"

masif_opts = {}
# Default directories
masif_opts["raw_pdb_dir"] = os.path.join(basedir, "data_preparation/00-raw_pdbs/")
masif_opts["pdb_chain_dir"] = os.path.join(basedir, "data_preparation/01-benchmark_pdbs/")
masif_opts["ply_chain_dir"] = os.path.join(basedir, "data_preparation/01-benchmark_surfaces/")
#masif_opts["tmp_dir"] = tempfile.gettempdir()
masif_opts["tmp_dir"] = '/data02/daniel/tmp'
masif_opts["ply_file_template"] = masif_opts["ply_chain_dir"] + "/{}_{}.ply"

# Surface features
masif_opts["use_hbond"] = True
masif_opts["use_hphob"] = True
masif_opts["use_apbs"] = True
masif_opts["compute_iface"] = True
# Mesh resolution. Everything gets very slow if it is lower than 1.0
masif_opts["mesh_res"] = 1.0
masif_opts["feature_interpolation"] = True

####
masif_opts['ligand_list'] = ["ADP", "COA", "FAD", "HEM", "NAD", "NAP", "SAM"]
masif_opts['solvents'] = ['IMP', 'EPE', 'FMN', 'TRS', 'PGE', 'ACT', 'NAG', 'EDO', 'GOL', 'SO4', 'PO4']
#masif_opts['solvents'] = ['EPE', 'FMN', 'TRS', 'PGE', 'NAG', 'ACT', 'EDO']
masif_opts['all_ligands'] = masif_opts['ligand_list'] + masif_opts['solvents']
####

# Coords params
masif_opts["radius"] = 12.0

# Neural network patch application specific parameters.
masif_opts["ppi_search"] = {}
masif_opts["ppi_search"]["training_list"] = "lists/training.txt"
masif_opts["ppi_search"]["testing_list"] = "lists/testing.txt"
masif_opts["ppi_search"]["max_shape_size"] = 200
masif_opts["ppi_search"]["max_distance"] = 12.0  # Radius for the neural network.
masif_opts["ppi_search"][
    "masif_precomputation_dir"
] = "/data02/daniel/masif/search/data_preparation/04b-precomputation_12A/precomputation/"
masif_opts["ppi_search"]["feat_mask"] = [1.0] * 5
masif_opts["ppi_search"]["max_sc_filt"] = 1.0
masif_opts["ppi_search"]["min_sc_filt"] = 0.5
masif_opts["ppi_search"]["pos_surf_accept_probability"] = 1.0
masif_opts["ppi_search"]["pos_interface_cutoff"] = 1.0
masif_opts["ppi_search"]["range_val_samples"] = 0.9  # 0.9 to 1.0
masif_opts["ppi_search"]["cache_dir"] = "nn_models/sc05/cache/"
masif_opts["ppi_search"]["model_dir"] = "nn_models/sc05/all_feat/model_data/"
masif_opts["ppi_search"]["desc_dir"] = "descriptors/sc05/all_feat/"
masif_opts["ppi_search"]["gif_descriptors_out"] = "gif_descriptors/"
# Parameters for shape complementarity calculations.
masif_opts["ppi_search"]["sc_radius"] = 12.0
masif_opts["ppi_search"]["sc_interaction_cutoff"] = 1.5
masif_opts["ppi_search"]["sc_w"] = 0.25

# Neural network patch application specific parameters.
masif_opts["site"] = {}
masif_opts["site"]["training_list"] = "lists/training.txt"
masif_opts["site"]["testing_list"] = "lists/testing.txt"
masif_opts["site"]["max_shape_size"] = 100
masif_opts["site"]["n_conv_layers"] = 3
masif_opts["site"]["max_distance"] = 9.0  # Radius for the neural network.
masif_opts["site"]["masif_precomputation_dir"] = "/data02/daniel/masif/site/data_preparation/04a-precomputation_9A/precomputation/"
#############
masif_opts["site"]["ligand_coords_dir"] = "/data02/daniel/masif/site/data_preparation/00c-ligand_coords/"
masif_opts["site"]["assembly_dir"] = "/data02/daniel/masif/site/data_preparation/00b-pdbs_assembly/"
masif_opts['site']["raw_pdb_dir"] = "/data02/daniel/masif/site/data_preparation/00-raw_pdbs/"
#############
masif_opts["site"]["range_val_samples"] = 0.9  # 0.9 to 1.0
masif_opts["site"]["model_dir"] = "nn_models/all_feat_3l/model_data/"
masif_opts["site"]["out_pred_dir"] = "output/all_feat_3l/pred_data/"
masif_opts["site"]["out_surf_dir"] = "output/all_feat_3l/pred_surfaces/"
masif_opts["site"]["feat_mask"] = [1.0] * 5

# Neural network ligand application specific parameters.
masif_opts["ligand"] = {}
masif_opts['ligand']["raw_pdb_dir"] = "/data02/daniel/masif/masif_ligand/data_preparation/00-raw_pdbs/"
masif_opts["ligand"]["assembly_dir"] = "/data02/daniel/masif/masif_ligand/data_preparation/00b-pdbs_assembly/"
masif_opts["ligand"]["ligand_coords_dir"] = "/data02/daniel/masif/masif_ligand/data_preparation/00c-ligand_coords/"
masif_opts["ligand"]["masif_precomputation_dir"] = "/data02/daniel/masif/masif_ligand/data_preparation/04a-precomputation_12A/precomputation/"
masif_opts["ligand"]["max_shape_size"] = 200
masif_opts["ligand"]["feat_mask"] = [1.0] * 5
masif_opts["ligand"]["train_fract"] = 0.9 * 0.8
masif_opts["ligand"]["val_fract"] = 0.1 * 0.8
masif_opts["ligand"]["test_fract"] = 0.2
masif_opts["ligand"]["tfrecords_dir"] = os.path.join(basedir, "data_preparation/tfrecords")
masif_opts["ligand"]["max_distance"] = 12.0
masif_opts["ligand"]["n_classes"] = 7
masif_opts["ligand"]["feat_mask"] = [1.0, 1.0, 1.0, 1.0, 1.0]
masif_opts["ligand"]["costfun"] = "dprime"
masif_opts["ligand"]["model_dir"] = "nn_models/all_feat/"
masif_opts["ligand"]["test_set_out_dir"] = "test_set_predictions/"

masif_opts['ligand']['minPockets'] = 32
masif_opts['ligand']['defaultCode'] = -1234567
masif_opts['ligand']['savedPockets'] = 200
masif_opts['ligand']['empty_pocket_ratio'] = 10
masif_opts['ligand']['ligand_list'] = masif_opts['ligand_list']



masif_opts['ligand_site'] = masif_opts['ligand'].copy()
masif_opts["ligand_site"]["max_shape_size"] = 100
masif_opts["ligand_site"]["n_conv_layers"] = 3
masif_opts["ligand_site"]["max_distance"] = 9.0  # Radius for the neural network.
masif_opts["ligand_site"]["masif_precomputation_dir"] = "/data02/daniel/masif/masif_ligand/data_preparation/04a-precomputation_9A/precomputation/"
masif_opts["ligand_site"]["training_list"] = "lists/training.txt"
masif_opts["ligand_site"]["testing_list"] = "lists/testing.txt"
masif_opts["ligand_site"]["range_val_samples"] = 0.9  # 0.9 to 1.0
masif_opts["ligand_site"]["model_dir"] = "nn_models/all_feat_3l/model_data/"
masif_opts["ligand_site"]["out_pred_dir"] = "output/all_feat_3l/pred_data/"
masif_opts["ligand_site"]["out_surf_dir"] = "output/all_feat_3l/pred_surfaces/"
masif_opts["ligand_site"]["feat_mask"] = [1.0] * 5

masif_opts['LSResNet'] = masif_opts['ligand_site'].copy()
masif_opts['LSResNet']['scale'] = 0.5
masif_opts['LSResNet']['max_dist'] = 35
