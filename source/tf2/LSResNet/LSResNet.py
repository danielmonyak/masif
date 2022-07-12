import tfbio.data
import numpy as np
from tensorflow.keras import layers, Sequential, initializers, Model
from tensorflow.keras.regularizers import L2
from tensorflow.keras import backend as K
import functools
from operator import add
from default_config.util import *
import math

params = masif_opts["ligand"]

def runLayers(layers, x):
    for l in layers:
        x = l(x)
    return x

class LSResNet(Model):
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
        n_conv_layers = 1,
        conv_batch_size = 1000
    ):
        ## Call super - model initializer
        super(LSResNet, self).__init__()
        
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
        self.conv_batch_size = conv_batch_size
        
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
        
        resolution = 1. / self.scale
        self.myMakeGrid = MakeGrid(max_dist=self.max_dist, grid_resolution=resolution)
        
        if K.image_data_format()=='channels_last':
            bn_axis=4
        else:
            bn_axis=1

        f=5
        filters = [f, f, f ]
        filters1,filters2,filters3=filters
        strides = (1,1,1)
        
        self.convBlock=[
            [layers.Conv3D(filters1, kernel_size=1, strides=strides, kernel_regularizer=L2(1e-4)),
            layers.BatchNormalization(axis=bn_axis),
            self.ReLU,
             
            layers.Conv3D(filters2, kernel_size=3, padding='same', kernel_regularizer=L2(1e-4)),
            layers.BatchNormalization(axis=bn_axis),
            self.ReLU,
             
            layers.Conv3D(filters3, kernel_size=1, kernel_regularizer=L2(1e-4)),
            layers.BatchNormalization(axis=bn_axis)],
            
            [layers.Conv3D(filters3, kernel_size=1, strides=strides, kernel_regularizer=L2(1e-4)),
            layers.BatchNormalization(axis=bn_axis)]
        ]
        
        self.lastConvLayer = layers.Conv3D(1, kernel_size=1, kernel_regularizer=L2(1e-4), activation='sigmoid')
    
    def runConv(self, x):
        n_pockets = tf.shape(x)[0]
        rg = range(0, n_pockets, self.conv_batch_size)
        ret_list = []
        for i in range(len(rg)-1):
            print(i)
            sample = tf.range(rg[i], rg[i+1])
            x_samp = tf.gather(x, sample, axis=0)
            ret = self.myConvLayer(x_samp)
            ret_list.append(ret)
        sample = tf.range(rg[-1], n_pockets)
        if sample.shape[0] != 0:
            x_samp = tf.gather(x, sample, axis=0)
            ret = self.myConvLayer(x_samp)
            ret_list.append(ret)
        return tf.concat(ret_list, axis=0)
        
    def call(self, X_packed, training=False):
        X, xyz_coords = X_packed
        
        ret = tf.map_fn(fn=self.runConv, elems = X, fn_output_signature = tf.TensorSpec(shape=[None, self.n_thetas * self.n_rhos * self.n_feat], dtype=tf.float32))
        #ret = self.myConvLayer(X)
        
        ret = runLayers(self.denseReduce, ret)
        
        
        '''ret = tfbio.data.make_grid(xyz_coords[0], ret[0],
                                   max_dist=self.max_dist,
                                   grid_resolution=resolution)'''
        ret = self.myMakeGrid(xyz_coords, ret)
        
        ret1 = runLayers(self.convBlock[0], ret)
        residue = runLayers(self.convBlock[1], ret)
        ret = tf.add(ret1, residue)
        ret = tf.nn.relu(ret)
        
        ret = self.lastConvLayer(ret)
        
        return ret
    '''
    def map_func(self, packed):
        resolution = 1. / self.scale
        
        xyz_coords = tf.gather(packed, tf.range(3), axis=-1)
        y_raw = tf.expand_dims(tf.gather(packed, 3, axis=-1), axis=-1)
        
        print(f'y_raw: {y_raw.shape}')
        print(f'xyz_coords: {xyz_coords.shape}')
        
        y = tfbio.data.make_grid(xyz_coords, y_raw, max_dist=self.max_dist, grid_resolution=resolution)
        return tf.squeeze(y, axis=0)
    
    def make_y(self, y_raw, xyz_coords):
        print(f'y_raw: {y_raw.shape}')
        print(f'xyz_coords: {xyz_coords.shape}')
        
        packed = tf.concat([xyz_coords, tf.cast(y_raw, dtype=tf.float64)], axis=-1)
        print(f'packed: {packed.shape}')
        
        return tf.map_fn(fn=self.map_func, elems = packed, fn_output_signature = tf.TensorSpec(shape=[36,36,36,1], dtype=tf.float32))
    
    def train_step(self, data):
        X_packed, y_raw = data
        
        xyz_coords = X_packed[1]
        y = self.make_y(y_raw, xyz_coords)
        
        with tf.GradientTape() as tape:
            y_pred = self(X_packed, training=True)  # Forward pass
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)
        
        return {m.name: m.result() for m in self.metrics}'''
    
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
    
    '''def call(self, x):
        return tf.map_fn(fn=self.call_wrapped, elems = x, fn_output_signature = tf.TensorSpec(shape=[None, self.n_thetas * self.n_rhos * self.n_feat],
                                                                                                   dtype=tf.float32))
        #return tf.map_fn(fn=self.call_wrapped, elems = x, fn_output_signature = tf.RaggedTensorSpec(shape=[None, self.n_thetas * self.n_rhos * self.n_feat],
        #                                                                                           dtype=tf.float32, ragged_rank=1, row_splits_dtype=tf.float32))
    '''
    def call(self, x):
        input_feat = tf.gather(x, tf.range(5), axis=-1)
        rho_coords = tf.gather(x, 5, axis=-1)
        theta_coords = tf.gather(x, 6, axis=-1)
        mask = tf.expand_dims(tf.gather(x, 7, axis=-1), axis=-1)
        #input_feat, rho_coords, theta_coords, mask = x

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
        n_samples = tf.shape(rho_coords)[0]
        n_vertices = tf.shape(rho_coords)[1]

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

class MakeGrid(layers.Layer):
    def __init__(self, max_dist=10.0, grid_resolution=1.0):
        if grid_resolution <= 0:
            raise ValueError('grid_resolution must be positive')
        if max_dist <= 0:
            raise ValueError('max_dist must be positive')
        
        self.grid_resolution = tf.constant(grid_resolution, dtype=tf.float32)
        self.max_dist = tf.constant(max_dist, dtype=tf.float32)
        self.box_size = tf.cast(tf.math.ceil(2 * max_dist / grid_resolution + 1), dtype=tf.int32)
        
        super(MakeGrid, self).__init__()
    def call(self, coords, features):
        c_shape = tf.shape(coords)
        N = c_shape[1]
        f_shape = features.shape
        '''
        print(c_shape)
        if tf.shape(c_shape) != 3 or c_shape[2] != 3:
            raise ValueError('coords must be an array of floats of shape (None, N, 3)')
        if tf.shape(f_shape) != 3 or f_shape[1] != N:
            raise ValueError('features must be an array of floats of shape (None, N, F)')'''
        
        batches = f_shape[0]
        num_features = f_shape[2]
        
        grid_coords = (coords + self.max_dist) / self.grid_resolution
        grid_coords = tf.cast(tf.round(grid_coords), dtype=tf.int32)

        in_box = tf.squeeze(tf.reduce_all(tf.logical_and(tf.greater_equal(grid_coords, 0), tf.less(grid_coords, self.box_size)), axis=2))
        grid_coords_IN = tf.boolean_mask(grid_coords, in_box, axis=1)
        features_IN = tf.boolean_mask(features, in_box, axis=1)
        
        #### Cannot handle multiple batches at once!!!!!!!!!!!!!!!!!!!!!
        
        idx = tf.concat([tf.zeros([batches, tf.shape(grid_coords_IN)[1], 1], dtype=tf.int32), grid_coords_IN], axis=-1)
        grid = tf.scatter_nd(indices=idx, updates=features_IN, shape=(batches, self.box_size, self.box_size, self.box_size, num_features))
        
        return grid
