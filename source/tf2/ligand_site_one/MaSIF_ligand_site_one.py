import numpy as np
from tensorflow.keras import layers, Sequential, Model
from default_config.util import *

params = masif_opts["ligand"]

class MaSIF_ligand_site(Model):
    """
    The neural network model.
    """ 
    def __init__(
        self,
        max_rho,
        n_ligands,
        n_thetas=16,
        n_rhos=5,
        learning_rate=1e-4,
        n_rotations=16,
        feat_mask=[1.0, 1.0, 1.0, 1.0],
        n_conv_layers = 1
    ):
        ## Call super - model initializer
        super(MaSIF_ligand_site, self).__init__()
        
        
        # order of the spectral filters
        self.max_rho = max_rho
        self.n_thetas = n_thetas
        self.n_rhos = n_rhos
        self.n_ligands = n_ligands
        self.sigma_rho_init = (
            max_rho / 8
        )  # in MoNet was 0.005 with max radius=0.04 (i.e. 8 times smaller)
        self.sigma_theta_init = 1.0  # 0.25
        self.n_rotations = n_rotations
        self.n_feat = int(sum(feat_mask))
        
        
        self.opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits = True)
 
        self.myLayers=[
            layers.Dense(1, activation="relu"),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(20, activation='relu'),
            layers.Dense(1)
        ]
    
    #@tf.autograph.experimental.do_not_convert
    def call(self, x, sample = None, training=False):
        for l in self.myLayers:
            ret = l(ret)
        return ret
