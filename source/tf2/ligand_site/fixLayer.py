import numpy as np
from tensorflow.keras import layers, Sequential, initializers, Model
import functools
from default_config.util import *

params = masif_opts["ligand"]
minPockets = params['minPockets']
prepSize = 2 * params['savedPockets']

class ConvLayer2(layers.Layer):
    def __init__(self,
        max_rho,
        n_ligands,
        n_thetas,
        n_rhos,
        n_rotations,
        feat_mask,
        n_conv_layers):
        
        super(ConvLayer, self).__init__()
        
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
        
        
        # Variable dict lists
        self.variable_dicts = []

        initial_coords = self.compute_initial_coordinates()
        # self.rotation_angles = tf.Variable(np.arange(0, 2*np.pi, 2*np.pi/self.n_rotations).astype('float32'))
        
        conv_shapes = [[self.n_thetas * self.n_rhos, self.n_thetas * self.n_rhos],
                       [self.n_feat * self.n_thetas * self.n_rhos, self.n_feat * self.n_thetas * self.n_rhos],
                       [self.n_feat * self.n_thetas * self.n_rhos, self.n_feat * self.n_thetas * self.n_rhos],
                       [self.n_thetas * self.n_rhos * self.n_thetas * self.n_rhos, self.n_thetas * self.n_rhos * self.n_thetas * self.n_rhos]]
        self.reshape_shapes = [[-1, minPockets, self.n_thetas * self.n_rhos * self.n_feat],
                               [-1, minPockets, self.n_feat, self.n_thetas * self.n_rhos],
                               [-1, minPockets, self.n_feat, self.n_thetas * self.n_rhos],
                               [-1, minPockets, self.n_thetas * self.n_rhos, self.n_thetas * self.n_rhos]]
        self.reduce_funcs = [None,
                             lambda x : tf.reduce_mean(x, axis=-1),
                             lambda x : tf.reduce_mean(x, axis=-1),
                             lambda x : tf.reduce_max(x, axis=-1)]
        
        mu_rho_initial = np.expand_dims(initial_coords[:, 0], 0).astype(
            "float32"
        )
        mu_theta_initial = np.expand_dims(initial_coords[:, 1], 0).astype(
            "float32"
        )
        mu_rho = []
        mu_theta = []
        sigma_rho = []
        sigma_theta = []

        b_conv = []
        W_conv = []
        ## mu_rho and mu_theta inital values are used for sigma as well -- check on this
        
        self.testVar = tf.Variable(3, name = 'testVar', trainable = True)
        self.testVar2 = self.add_weight(name = 'testVar2', initializer='zeros', trainable = True)
        
        layer_num = 0
        for i in range(self.n_feat):
            mu_rho.append(
                tf.Variable(mu_rho_initial, name="mu_rho_{}_{}".format(i, layer_num),
                           trainable = True)
            )  # 1, n_gauss
            mu_theta.append(
                tf.Variable(mu_theta_initial, name="mu_theta_{}_{}".format(i, layer_num),
                           trainable = True)
            )  # 1, n_gauss
            sigma_rho.append(
                tf.Variable(
                    np.ones_like(mu_rho_initial) * self.sigma_rho_init,
                    name="sigma_rho_{}_{}".format(i, layer_num),
                    trainable = True
                )
            )  # 1, n_gauss
            sigma_theta.append(
                tf.Variable(
                    (np.ones_like(mu_theta_initial) * self.sigma_theta_init),
                    name="sigma_theta_{}_{}".format(i, layer_num),
                    trainable = True
                )
            )  # 1, n_gauss


            b_conv.append(
                self.add_weight(
                    "b_conv_{}_{}".format(i, layer_num),
                    shape=conv_shapes[layer_num][1], initializer='zeros',
                    trainable = True
                )
            )
            W_conv.append(
                self.add_weight(
                    "W_conv_{}_{}".format(i, layer_num),
                    shape=conv_shapes[layer_num], initializer=initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                    trainable = True
                )
            )
            
        
        var_dict = {}
        var_dict['mu_rho'] = mu_rho
        var_dict['mu_theta'] = mu_theta
        var_dict['sigma_rho'] = sigma_rho
        var_dict['sigma_theta'] = sigma_theta
        var_dict['b_conv'] = b_conv
        var_dict['W_conv'] = W_conv
        
        self.variable_dicts.append(var_dict)
    
    def compute_initial_coordinates(self):
        range_rho = [0.0, self.max_rho]
        range_theta = [0, 2 * np.pi]

        grid_rho = np.linspace(range_rho[0], range_rho[1], num=self.n_rhos + 1)
        grid_rho = grid_rho[1:]
        grid_theta = np.linspace(range_theta[0], range_theta[1], num=self.n_thetas + 1)
        grid_theta = grid_theta[:-1]

        grid_rho_, grid_theta_ = np.meshgrid(grid_rho, grid_theta, sparse=False)
        grid_rho_ = (
            grid_rho_.T
        )  # the traspose here is needed to have the same behaviour as Matlab code
        grid_theta_ = (
            grid_theta_.T
        )  # the traspose here is needed to have the same behaviour as Matlab code
        grid_rho_ = grid_rho_.flatten()
        grid_theta_ = grid_theta_.flatten()

        coords = np.concatenate((grid_rho_[None, :], grid_theta_[None, :]), axis=0)
        coords = coords.T  # every row contains the coordinates of a grid intersection
        return coords
