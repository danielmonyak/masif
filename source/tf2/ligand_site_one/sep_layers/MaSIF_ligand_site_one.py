import numpy as np
from tensorflow.keras import layers, Sequential, initializers, Model, regularizers
import tensorflow as tf
import functools
from operator import add
from default_config.util import *

params = masif_opts["ligand"]

def runLayers(layers, x):
    for l in layers:
        x = l(x)
    return x

class MaSIF_ligand_site(Model):
    """
    The neural network model.
    """ 
    def __init__(
        self,
        max_rho,
        n_thetas=16,
        n_rhos=5,
        learning_rate=1e-4,
        n_rotations=16,
        feat_mask=[1.0, 1.0, 1.0, 1.0],
        keep_prob = 1.0,
        reg_val = 1e-4,
        reg_type = 'l2'
    ):
        ## Call super - model initializer
        super(MaSIF_ligand_site, self).__init__()
        
        regKwargs = {reg_type : reg_val}
        reg = regularizers.L1L2(**regKwargs)
        
        # order of the spectral filters
        self.max_rho = max_rho
        self.n_thetas = n_thetas
        self.n_rhos = n_rhos
        self.sigma_rho_init = (
            max_rho / 8
        )  # in MoNet was 0.005 with max radius=0.04 (i.e. 8 times smaller)
        self.sigma_theta_init = 1.0  # 0.25
        self.n_rotations = n_rotations
        self.n_feat = int(sum(feat_mask))
        
        self.opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits = True)
        
        self.conv_shapes = [[self.n_thetas * self.n_rhos, self.n_thetas * self.n_rhos],
                       [self.n_feat * self.n_thetas * self.n_rhos, self.n_feat * self.n_thetas * self.n_rhos],
                       [self.n_feat * self.n_thetas * self.n_rhos, self.n_feat * self.n_thetas * self.n_rhos],
                       [self.n_thetas * self.n_rhos * self.n_thetas * self.n_rhos, self.n_thetas * self.n_rhos * self.n_thetas * self.n_rhos]] 
        self.reshape_shapes = [[-1, self.n_thetas * self.n_rhos * self.n_feat],
                          [-1, self.n_feat, self.n_thetas * self.n_rhos],
                          [-1, self.n_feat, self.n_thetas * self.n_rhos],
                          [-1, self.n_thetas * self.n_rhos, self.n_thetas * self.n_rhos]]


        self.convBlock_arr = [
            [
                ConvLayer(0, self.conv_shapes[0], max_rho, n_thetas, n_rhos, n_rotations, feat_mask, reg),
                layers.Reshape(self.reshape_shapes[0]),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.Dense(self.n_thetas * self.n_rhos, activation="relu", kernel_regularizer=reg),
                layers.Dense(self.n_feat, activation="relu", kernel_regularizer=reg)
            ],
            [
                ConvLayer(1, self.conv_shapes[1], max_rho, n_thetas, n_rhos, n_rotations, feat_mask, reg),
                layers.Reshape(self.reshape_shapes[1]),
                MeanAxis1(out_shp=self.reshape_shapes[1][:2]),
                layers.BatchNormalization(),
                layers.ReLU()
            ],
            [
                ConvLayer(2, self.conv_shapes[2], max_rho, n_thetas, n_rhos, n_rotations, feat_mask, reg),
                layers.Reshape(self.reshape_shapes[2]),
                MeanAxis1(out_shp=self.reshape_shapes[2][:2]),
                layers.BatchNormalization(),
                layers.ReLU()
            ]
        ]
        self.myDense = layers.Dense(self.n_thetas, activation="relu", kernel_regularizer=reg)
        self.outLayer = layers.Dense(1, kernel_regularizer=reg)
        
    def call(self, x, training=False):
        data_tsrs, indices_tensor = x
        _, rho_coords, theta_coords, mask = data_tsrs
        
        ret = runLayers(self.convBlock_arr[0], data_tsrs)
        
        input_feat = tf.gather(ret, indices_tensor, batch_dims=1)
        ret = runLayers(self.convBlock_arr[0], (input_feat, rho_coords, theta_coords, mask))
        
        input_feat = tf.gather(ret, indices_tensor)
        ret = runLayers(self.convBlock_arr[0], (input_feat, rho_coords, theta_coords, mask))
        
        ret = self.myDense(ret)
        ret = self.outLayer(ret)
        return ret

class MeanAxis1(layers.Layer):
    def __init__(self, out_shp):
        super(MeanAxis1, self).__init__()
        self.out_shp = out_shp
    def callInner(self, x):
        return tf.reduce_mean(x, axis=-1)
    def call(self, x):
        return tf.map_fn(fn=self.callInner, elems = x, fn_output_signature = tf.TensorSpec(shape=self.out_shp, dtype=tf.float32))
    
class ConvLayer(layers.Layer):
    def __init__(self,
        layer_num,
        conv_shape,
        max_rho,
        n_thetas,
        n_rhos,
        n_rotations,
        feat_mask,
        reg=None):
        
        super(ConvLayer, self).__init__()
        
        # order of the spectral filters
        self.max_rho = max_rho
        self.n_thetas = n_thetas
        self.n_rhos = n_rhos
        self.sigma_rho_init = (
            max_rho / 8
        )  # in MoNet was 0.005 with max radius=0.04 (i.e. 8 times smaller)
        self.sigma_theta_init = 1.0  # 0.25
        self.n_rotations = n_rotations
        self.n_feat = int(sum(feat_mask))
        
        initial_coords = self.compute_initial_coordinates()
        # self.rotation_angles = tf.Variable(np.arange(0, 2*np.pi, 2*np.pi/self.n_rotations).astype('float32'))
        
        mu_rho_initial = tf.cast(tf.expand_dims(initial_coords[:, 0], axis=0), dtype=tf.float32)
        mu_theta_initial = tf.cast(tf.expand_dims(initial_coords[:, 1], axis=0), dtype=tf.float32)
        
        self.mu_rho = []
        self.mu_theta = []
        self.sigma_rho = []
        self.sigma_theta = []

        self.b_conv = []
        self.W_conv = []
        
        self.conv_shape = conv_shape
        self.weights_num = [self.n_feat, 1, 1, 1][layer_num]
        
        for i in range(self.weights_num):
            self.mu_rho.append(
                self.add_weight(shape=tf.shape(mu_rho_initial),
                                initializer = ValueInit(mu_rho_initial), trainable = True, regularizer=reg)
            )  # 1, n_gauss
            self.mu_theta.append(
                self.add_weight(shape=tf.shape(mu_theta_initial),
                                initializer = ValueInit(mu_theta_initial), trainable = True, regularizer=reg)
            )  # 1, n_gauss
            self.sigma_rho.append(
                self.add_weight(shape=tf.shape(mu_rho_initial),
                                initializer = initializers.Constant(self.sigma_rho_init), trainable = True, regularizer=reg)
            )  # 1, n_gauss
            self.sigma_theta.append(
                self.add_weight(shape=tf.shape(mu_theta_initial),
                                initializer = initializers.Constant(self.sigma_theta_init), trainable = True, regularizer=reg)
            )  # 1, n_gauss

            self.b_conv.append(
                self.add_weight(
                    shape=self.conv_shape[1], initializer='zeros',
                    trainable = True
                )
            )
            self.W_conv.append(
                self.add_weight(
                    shape=self.conv_shape, initializer=initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                    trainable = True, regularizer=reg
                )
            )
    
    def callInner(self, x):
        input_feat, rho_coords, theta_coords, mask = [tf.cast(tsr, dtype=tf.float32) for tsr in x]
        
        ret = []
        for i in range(self.weights_num):
            if self.weights_num == self.n_feat:
                my_input_feat = tf.gather(input_feat, tf.range(i, i+1), axis=-1)
            else:
                my_input_feat = input_feat
            
            ret.append(self.inference(
                my_input_feat,
                rho_coords,
                theta_coords,
                mask,
                self.W_conv[i],
                self.b_conv[i],
                self.mu_rho[i],
                self.sigma_rho[i],
                self.mu_theta[i],
                self.sigma_theta[i]
            ))
        
        ret = tf.stack(ret, axis=1)
        #ret = tf.stack(ret, axis=2)
        #return tf.squeeze(ret)
        return ret
    
    def call(self, data_tsrs):
        n_samples = data_tsrs[0].shape[1]
        out_shp = [n_samples, self.weights_num, self.conv_shape[0]]
        return tf.map_fn(fn=self.callInner, elems = data_tsrs, fn_output_signature = tf.TensorSpec(shape=out_shp, dtype=tf.float32))
    
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
        n_samples = tf.shape(rho_coords)[0]
        n_vertices = tf.shape(rho_coords)[1]
        n_feat = tf.shape(input_feat)[2]
        
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

            # check the axis on this
            if mean_gauss_activation:  # computes mean weights for the different gaussians
                gauss_activations /= (
                    tf.reduce_sum(input_tensor=gauss_activations, axis=1, keepdims=True) + eps
                )  # batch_size, n_vertices, n_gauss

            gauss_activations = tf.expand_dims(
                gauss_activations, 2
            )  # batch_size, n_vertices, 1, n_gauss,

            input_feat_ = tf.expand_dims(
                input_feat, -1
            )  # batch_size, n_vertices, n_feat, 1


            gauss_desc = tf.multiply(
                gauss_activations, input_feat_
            )  # batch_size, n_vertices, n_feat, n_gauss,
            gauss_desc = tf.reduce_sum(gauss_desc, axis=1)  # batch_size, n_feat, n_gauss,
            gauss_desc = tf.reshape(
                gauss_desc, [n_samples, self.n_thetas * self.n_rhos * n_feat]
            ) # batch_size, self.n_thetas*self.n_rhos*n_feat
            

            conv_feat = tf.matmul(gauss_desc, W_conv) + b_conv  # batch_size, n_gauss
            all_conv_feat.append(conv_feat)
            
        all_conv_feat = tf.stack(all_conv_feat)
        conv_feat = tf.reduce_max(all_conv_feat, axis=0)
        
        #conv_feat = tf.nn.relu(conv_feat)
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
