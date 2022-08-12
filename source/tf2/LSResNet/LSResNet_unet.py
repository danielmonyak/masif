import numpy as np
from tensorflow.keras import layers, Sequential, initializers, Model, regularizers
import tensorflow as tf
from tensorflow.keras import backend as K
import default_config.util as util
from default_config.masif_opts import masif_opts

params = masif_opts['LSResNet']

class LSResNet(Model):
    def __init__(
        self,
        max_rho,
        n_thetas=16,
        n_rhos=5,
        n_rotations=16,
        feat_mask=[1.0, 1.0, 1.0, 1.0],
        reg_val = 1e-4,
        reg_type = 'l2'
    ):
        ## Call super - model initializer
        super(LSResNet, self).__init__()
        
        n_feat = int(sum(feat_mask))
        
        ##
        regKwargs = {reg_type : reg_val}
        reg = regularizers.L1L2(**regKwargs)
        ##
        
        conv_shape = [n_thetas * n_rhos, n_thetas * n_rhos]
        reshape_shape = [-1, n_thetas * n_rhos * n_feat]
        
        self.convBlock0 = [
            ConvLayer(5, conv_shape, max_rho, n_thetas, n_rhos, n_rotations, n_feat, reg),
            layers.Reshape(reshape_shape)
        ]
        
        self.denseReduce = [
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dense(n_thetas * n_rhos, activation="relu"),
            layers.Dense(n_feat, activation="relu"),
        ]
        
        resolution = 1. / params['scale']
        self.myMakeGrid = MakeGrid(max_dist=params['max_dist'], grid_resolution=resolution)
        
        #####################################
        #####################################
        f=5
        
        self.CB_a2 = ConvBlock([f, f, f ], strides=(1,1,1))
        self.IB_b2 = IdentityBlock([f, f, f ])
        self.IB_c2 = IdentityBlock([f, f, f ])
        
        self.CB_a4 = ConvBlock([f*2, f*2, f * 2], strides=(2,2,2))
        self.IB_b4 = IdentityBlock([f*2,f*2,f * 2])
        self.IB_f4 = IdentityBlock([f*2,f*2,f * 2])
        
        self.CB_a5 = ConvBlock([f*4, f*4, f * 4], strides=(2,2,2))
        self.IB_b5 = IdentityBlock([f*4, f*4, f *4])
        self.IB_c5 = IdentityBlock([f*4, f*4, f * 4])
        
        self.CB_a6 = ConvBlock([f*8, f*8, f *8], strides=(3,3,3))
        self.IB_b6 = IdentityBlock([f*8, f*8, f *8])
        self.IB_c6 = IdentityBlock([f*8, f*8, f * 8])
        
        self.CB_a7 = ConvBlock([f*16, f*16, f *16], strides=(3,3,3))
        self.IB_b7 = IdentityBlock([f*16, f*16, f *16])
        
        self.UCB_a8 = UpConvBlock([f * 16, f * 16, f * 16], size=(3,3,3), padding='same')
        self.IB_b8 = IdentityBlock([f * 16, f * 16, f * 16])

        self.UCB_a9 = UpConvBlock([f * 8, f * 8, f * 8], size=(3,3,3), stride=(1,1,1))
        self.IB_b9 = IdentityBlock([f * 8, f * 8, f * 8])

        self.UCB_a10 = UpConvBlock([f * 4, f*4 , f*4 ], size=(2,2,2), stride=(1,1,1))
        self.IB_b10 = IdentityBlock([f * 4, f*4 , f*4 ])

        self.UCB_a11 = UpConvBlock([f*2 , f*2 , f*2 ], size=(2,2,2), stride=(1,1,1))
        self.IB_b11 = IdentityBlock([f*2 , f*2 , f*2 ])
        #####################################
        #####################################
        
        self.lastConvLayer = layers.Conv3D(filters=1, kernel_size=1)
        
    def call(self, X_packed, training=False):
        X, xyz_coords = X_packed
        
        ret = util.runLayers(self.convBlock0, X)
        ret = util.runLayers(self.denseReduce, ret)
        
        ret = self.myMakeGrid(xyz_coords, ret)
        
        ret = self.run_UNet(ret)
        ret = self.lastConvLayer(ret)
        return ret

    def run_UNet(self, inputs):
        x = self.CB_a2(inputs)
        x = self.IB_b2(x)
        x1 = self.IB_c2(x)
        
        x = self.CB_a4(x)
        x = self.IB_b4(x)
        x2 = self.IB_f4(x)
        
        x = self.CB_a5(x)
        x = self.IB_b5(x)
        x3 = self.IB_c5(x)
        
        x = self.CB_a6(x)
        x = self.IB_b6(x)
        x4 = self.IB_c6(x)
        
        x = self.CB_a7(x)
        x = self.IB_b7(x)
        
        x = self.UCB_a8(x)
        x = self.IB_b8(x)

        x = tf.concat([x, x4], axis=4)
        x = self.UCB_a9(x)
        x = self.IB_b9(x)

        x = tf.concat([x, x3], axis=4)
        x = self.UCB_a10(x)
        x = self.IB_b10(x)
        
        x = tf.concat([x, x2], axis=4)
        x = self.UCB_a11(x)
        x = self.IB_b11(x)
    
        x = tf.concat([x, x1], axis=4)
        return x
    
class ConvLayer(layers.Layer):
    def __init__(self,
        weights_num,
        conv_shape,
        max_rho,
        n_thetas,
        n_rhos,
        n_rotations,
        n_feat,
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
        self.n_feat = n_feat
        
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
        #self.weights_num = [self.n_feat, 1, 1, 1][layer_num]
        #if layer_num > 0:
        self.weights_num = weights_num
        
        for i in range(self.weights_num):
            self.mu_rho.append(
                self.add_weight("mu_rho_{}".format(i), shape=tf.shape(mu_rho_initial),
                                initializer = util.ValueInit(mu_rho_initial), trainable = True, regularizer=reg)
            )  # 1, n_gauss
            self.mu_theta.append(
                self.add_weight("mu_theta_{}".format(i), shape=tf.shape(mu_theta_initial),
                                initializer = util.ValueInit(mu_theta_initial), trainable = True, regularizer=reg)
            )  # 1, n_gauss
            self.sigma_rho.append(
                self.add_weight("sigma_rho_{}".format(i), shape=tf.shape(mu_rho_initial),
                                initializer = initializers.Constant(self.sigma_rho_init), trainable = True, regularizer=reg)
            )  # 1, n_gauss
            self.sigma_theta.append(
                self.add_weight("sigma_theta_{}".format(i), shape=tf.shape(mu_theta_initial),
                                initializer = initializers.Constant(self.sigma_theta_init), trainable = True, regularizer=reg)
            )  # 1, n_gauss

            self.b_conv.append(
                self.add_weight("b_conv_{}".format(i), shape=self.conv_shape[1], initializer='zeros', trainable = True)
            )
            self.W_conv.append(
                self.add_weight("W_conv_{}".format(i), shape=self.conv_shape, initializer=initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), trainable = True, regularizer=reg)
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
        batches = tf.shape(coords)[0]
        c_shape = coords.shape
        f_shape = features.shape
        N = c_shape[1]
        num_features = f_shape[2]
        
        grid_coords = (coords + self.max_dist) / self.grid_resolution
        grid_coords = tf.cast(tf.round(grid_coords), dtype=tf.int32)

        in_box = tf.reduce_all(tf.logical_and(tf.greater_equal(grid_coords, 0), tf.less(grid_coords, self.box_size)), axis=2)
        grid_coords_IN = tf.boolean_mask(grid_coords, in_box)
        features_IN = tf.boolean_mask(features, in_box)
        
        #### Cannot handle multiple batches at once!!!!!!!!!!!!!!!!!!!!!
        
        idx = tf.concat([tf.zeros([tf.shape(grid_coords_IN)[0], 1], dtype=tf.int32), grid_coords_IN], axis=-1)
        grid = tf.scatter_nd(indices=idx, updates=features_IN, shape=(batches, self.box_size, self.box_size, self.box_size, num_features))
        
        return grid

class ConvBlock:
    def __init__(self, filters, strides=(2,2,2)):
        filters1,filters2,filters3=filters
        
        if K.image_data_format()=='channels_last':
            bn_axis=4
        else:
            bn_axis=1
        
        self.mainBlock = [
            layers.Conv3D(filters1, kernel_size=1, strides=strides),
            layers.BatchNormalization(axis=bn_axis),
            layers.ReLU(),
             
            layers.Conv3D(filters2, kernel_size=3, padding='same'),
            layers.BatchNormalization(axis=bn_axis),
            layers.ReLU(),
             
            layers.Conv3D(filters3, kernel_size=1),
            layers.BatchNormalization(axis=bn_axis)
        ]
        
        self.resBlock = [
            layers.Conv3D(filters3, kernel_size=1, strides=strides),
            layers.BatchNormalization(axis=bn_axis)
        ]
        
    def __call__(self, x):
        ret = util.runLayers(self.mainBlock, x)
        residue = util.runLayers(self.resBlock, x)
        ret = tf.add(ret, residue)
        ret = tf.nn.relu(ret)
        return ret
        
        
class IdentityBlock:
    def __init__(self, filters, layer=None):
        filters1,filters2,filters3=filters
        
        if K.image_data_format()=='channels_last':
            bn_axis=4
        else:
            bn_axis=1
        
        self.mainBlock = []
        self.mainBlock.append(layers.Conv3D(filters1, kernel_size=1))
        if layer is None:
            self.mainBlock.append(layers.BatchNormalization(axis=bn_axis))
        self.mainBlock.append(layers.ReLU())
             
        self.mainBlock.append(layers.Conv3D(filters2, kernel_size=3, padding='same'))
        if layer is None:
            self.mainBlock.append(layers.BatchNormalization(axis=bn_axis))
        self.mainBlock.append(layers.ReLU())
             
        self.mainBlock.append(layers.Conv3D(filters3, kernel_size=1))
        if layer is None:
            self.mainBlock.append(layers.BatchNormalization(axis=bn_axis))
        
        
    def __call__(self, x):
        ret = util.runLayers(self.mainBlock, x)
        ret = tf.add(ret, x)
        ret = tf.nn.relu(ret)
        return ret

class UpConvBlock:
    def __init__(self, filters, stride=(1,1,1), size=(2,2,2), padding='same', layer=None):
        filters1,filters2,filters3=filters
        
        if K.image_data_format()=='channels_last':
            bn_axis=4
        else:
            bn_axis=1
    
        self.mainBlock = []
        self.mainBlock.append(layers.UpSampling3D(size))
        self.mainBlock.append(layers.Conv3D(filters1, kernel_size=1, strides=stride))
        if layer is None:
            self.mainBlock.append(layers.BatchNormalization(axis=bn_axis))
        self.mainBlock.append(layers.ReLU())

        self.mainBlock.append(layers.Conv3D(filters2, kernel_size=3, padding=padding, strides=stride))
        if layer is None:
            self.mainBlock.append(layers.BatchNormalization(axis=bn_axis))
        self.mainBlock.append(layers.ReLU())

        self.mainBlock.append(layers.Conv3D(filters3, kernel_size=1))
        if layer is None:
            self.mainBlock.append(layers.BatchNormalization(axis=bn_axis))


        self.resBlock = []
        self.resBlock.append(layers.UpSampling3D(size))
        self.resBlock.append(layers.Conv3D(filters3, kernel_size=1, strides=stride, padding=padding))
        if layer is None:
            self.resBlock.append(layers.BatchNormalization(axis=bn_axis))
    
    def __call__(self, x):
        ret = util.runLayers(self.mainBlock, x)
        residue = util.runLayers(self.resBlock, x)
        ret = tf.add(ret, residue)
        ret = tf.nn.relu(ret)
        return ret
