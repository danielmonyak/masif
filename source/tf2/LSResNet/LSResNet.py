import numpy as np
from tensorflow.keras import layers, Sequential, initializers, Model
from tensorflow.keras.regularizers import L2
from tensorflow.keras import backend as K
import functools
from operator import add
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
        keep_prob = 1.0,
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
        
        self.scale = 0.5
        self.max_dist = 35
        
        self.opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits = True)
 
        self.ReLU = layers.ReLU()
        self.Add = layers.Add()
        
        self.myConvLayer = ConvLayer(max_rho, n_ligands, n_thetas, n_rhos, n_rotations, feat_mask, n_conv_layers)
        self.denseReduce = [
            layers.BatchNormalization(),
            self.ReLU,
            layers.Dense(self.n_thetas * self.n_rhos, activation="relu"),
            layers.Dense(self.n_feat, activation="relu"),
        ]
        
        if K.image_data_format()=='channels_last':
            bn_axis=4
        else:
            bn_axis=1

        f=18
        filters = [f, f, f ]
        filters1,filters2,filters3=filters
        
        self.convBlock=[
            [layers.Conv3D(filters1, kernel_size=1, strides=strides, kernel_regularizer=L2(1e-4)),
            layers.BatchNormalization(axis=bn_axis),
            self.ReLU,
             
            layers.Conv3D(filters2, kernel_size=3 ,padding='same', kernel_regularizer=L2(1e-4)),
            layers.BatchNormalization(axis=bn_axis),
            self.ReLU,
             
            layers.Conv3D(filters3, kernel_size=1, kernel_regularizer=L2(1e-4)),
            layers.BatchNormalization(axis=bn_axis)],
            
            [layers.Conv3D(filters3, kernel_size=1, strides=strides, kernel_regularizer=L2(1e-4)),
            layers.BatchNormalization(axis=bn_axis)]
        ]
    
    #@tf.autograph.experimental.do_not_convert
    def call(self, x, training=False):
        xyz_coords = x[1]
        ret = self.myConvLayer(x[0])
        for l in self.denseReduce:
            ret = l(ret)
        
        resolution = 1. / self.scale
        ret = tfbio.data.make_grid(xyz_coords, ret,
                                 max_dist=self.max_dist,
                                 grid_resolution=resolution)
        
        ret1 = self.convBlock[0](ret)
        residue = self.convBlock[1](ret)
        ret = self.Add([ret1, residue])
        ret = self.Relu(ret)
        
        return ret

class ConvLayer(layers.Layer):
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
        
        ####
        #self.bigShape = [200, self.n_feat]
        #self.smallShape = [200, 1]
        #self.bigIdx = tf.cast(functools.reduce(prodFunc, bigShape), dtype = tf.int32)
        ####
        
        # Variable dict lists
        self.variable_dicts = []
        self.relu = tf.keras.layers.ReLU()
        
        initial_coords = self.compute_initial_coordinates()
        # self.rotation_angles = tf.Variable(np.arange(0, 2*np.pi, 2*np.pi/self.n_rotations).astype('float32'))
        
        conv_shapes = [[self.n_thetas * self.n_rhos, self.n_thetas * self.n_rhos],
                       [self.n_feat * self.n_thetas * self.n_rhos, self.n_feat * self.n_thetas * self.n_rhos],
                       [self.n_feat * self.n_thetas * self.n_rhos, self.n_feat * self.n_thetas * self.n_rhos],
                       [self.n_thetas * self.n_rhos * self.n_thetas * self.n_rhos, self.n_thetas * self.n_rhos * self.n_thetas * self.n_rhos]]
        self.reshape_shapes = [[-1, self.n_thetas * self.n_rhos * self.n_feat],
                               [-1, self.n_feat, self.n_thetas * self.n_rhos],
                               [-1, self.n_feat, self.n_thetas * self.n_rhos],
                               [-1, self.n_thetas * self.n_rhos, self.n_thetas * self.n_rhos]]
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
        self.mu_rho = []
        self.mu_theta = []
        self.sigma_rho = []
        self.sigma_theta = []

        self.b_conv = []
        self.W_conv = []
        
        layer_num = 0
        for i in range(self.n_feat):
            self.mu_rho.append(
                #tf.Variable(mu_rho_initial, name="mu_rho_{}_{}".format(i, layer_num), trainable = True)
                self.add_weight(name="mu_rho_{}_{}".format(i, layer_num), shape=tf.shape(mu_rho_initial),
                                initializer = ValueInit(mu_rho_initial), trainable = True)
            )  # 1, n_gauss
            self.mu_theta.append(
                #tf.Variable(mu_theta_initial, name="mu_theta_{}_{}".format(i, layer_num), trainable = True)
                self.add_weight(name="mu_theta_{}_{}".format(i, layer_num), shape=tf.shape(mu_theta_initial),
                                initializer = ValueInit(mu_theta_initial), trainable = True)
            )  # 1, n_gauss
            self.sigma_rho.append(
                #tf.Variable(np.ones_like(mu_rho_initial) * self.sigma_rho_init, name="sigma_rho_{}_{}".format(i, layer_num), trainable = True)
                self.add_weight(name="sigma_rho_{}_{}".format(i, layer_num), shape=tf.shape(mu_rho_initial),
                                initializer = initializers.Constant(self.sigma_rho_init), trainable = True)
            )  # 1, n_gauss
            self.sigma_theta.append(
                #tf.Variable(np.ones_like(mu_theta_initial) * self.sigma_theta_init, name="sigma_theta_{}_{}".format(i, layer_num), trainable = True)
                self.add_weight(name="sigma_theta_{}_{}".format(i, layer_num), shape=tf.shape(mu_theta_initial),
                                initializer = initializers.Constant(self.sigma_theta_init), trainable = True)
            )  # 1, n_gauss


            self.b_conv.append(
                self.add_weight(
                    "b_conv_{}_{}".format(i, layer_num),
                    shape=conv_shapes[layer_num][1], initializer='zeros',
                    trainable = True
                )
            )
            self.W_conv.append(
                self.add_weight(
                    "W_conv_{}_{}".format(i, layer_num),
                    shape=conv_shapes[layer_num], initializer=initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                    trainable = True
                )
            )
    
    def call(self, x):
        input_feat, rho_coords, theta_coords, mask = x

        ret = []
        for i in range(self.n_feat):
            my_input_feat = tf.gather(input_feat, tf.range(i, i+1), axis=-1)
            ret.append(
                self.inference(
                    my_input_feat,
                    rho_coords,
                    theta_coords,
                    mask,
                    self.W_conv[i],
                    self.b_conv[i],
                    self.mu_rho[i],
                    self.sigma_rho[i],
                    self.mu_theta[i],
                    self.sigma_theta[i],
                )
            )  # batch_size, n_gauss*1

        ret = tf.stack(ret, axis=2)
        ret = tf.reshape(ret, self.reshape_shapes[0])
        return ret
        
    
    def inference(
        self,
        input_feat,
        rho_coords,
        theta_coords,
        mask,
        W_conv,
        b_conv,
        mu_rho,
        sigma_rho,
        mu_theta,
        sigma_theta,
        eps=1e-5,
        mean_gauss_activation=True,
    ):
        n_samples = tf.shape(input=rho_coords)[0]
        n_vertices = tf.shape(input=rho_coords)[1]

        all_conv_feat = []
        for k in range(self.n_rotations):

            rho_coords_ = tf.reshape(rho_coords, [-1, 1])  # batch_size*n_vertices
            thetas_coords_ = tf.reshape(theta_coords, [-1, 1])  # batch_size*n_vertices

            thetas_coords_ += k * 2 * np.pi / self.n_rotations
            thetas_coords_ = tf.math.mod(thetas_coords_, 2 * np.pi)
            rho_coords_ = tf.exp(
                -tf.square(rho_coords_ - mu_rho) / (tf.square(sigma_rho) + eps)
            )
            thetas_coords_ = tf.exp(
                -tf.square(thetas_coords_ - mu_theta) / (tf.square(sigma_theta) + eps)
            )

            gauss_activations = tf.multiply(
                rho_coords_, thetas_coords_
            )  # batch_size*n_vertices, n_gauss
            gauss_activations = tf.reshape(
                gauss_activations, [n_samples, n_vertices, -1]
            )  # batch_size, n_vertices, n_gauss
            gauss_activations = tf.multiply(gauss_activations, mask)

            if mean_gauss_activation:  # computes mean weights for the different gaussians
                gauss_activations /= (
                    tf.reduce_sum(input_tensor=gauss_activations, axis=2, keepdims=True) + eps
                )  # batch_size, n_vertices, n_gauss

            gauss_activations = tf.expand_dims(
                gauss_activations, 2
            )  # batch_size, n_vertices, 1, n_gauss,

            # check the axis on this
            input_feat_ = tf.expand_dims(
                input_feat, -1
            )  # batch_size, n_vertices, n_feat, 1


            gauss_desc = tf.multiply(
                gauss_activations, input_feat_
            )  # batch_size, n_vertices, n_feat, n_gauss,
            gauss_desc = tf.reduce_sum(gauss_desc, axis=1)  # batch_size, n_feat, n_gauss,
            gauss_desc = tf.reshape(
                gauss_desc, [n_samples, self.n_thetas * self.n_rhos]
            )  # batch_size, 80

            conv_feat = tf.matmul(gauss_desc, W_conv) + b_conv  # batch_size, 80
            all_conv_feat.append(conv_feat)
        all_conv_feat = tf.stack(all_conv_feat)
        conv_feat = tf.reduce_max(all_conv_feat, axis=0)
        
        conv_feat = tf.nn.relu(conv_feat)
        return conv_feat

    
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
