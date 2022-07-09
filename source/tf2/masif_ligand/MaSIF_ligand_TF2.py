import numpy as np
import functools
import tensorflow as tf
from tensorflow.keras import layers, Sequential, initializers, Model, regularizers
from default_config.util import *

#tf.debugging.set_log_device_placement(True)
params = masif_opts["ligand"]
minPockets = params['minPockets']


class MaSIF_ligand(Model):
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
        keep_prob = 1.0
    ):
        ## Call super - model initializer
        super(MaSIF_ligand, self).__init__()
        
        ##
        self.keep_prob = keep_prob
        ##
        
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
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
 
        
        self.myConvLayer = ConvLayer(max_rho, n_ligands, n_thetas, n_rhos, n_rotations, feat_mask)
        
        self.myLayers=[
            layers.Reshape([minPockets, self.n_feat * self.n_thetas * self.n_rhos]),
            #
            layers.BatchNormalization(),
            #
            layers.ReLU(),
            layers.Dense(self.n_thetas * self.n_rhos),
            #
            layers.BatchNormalization(),
            layers.ReLU(),
            #
            CovarLayer(),
            layers.Flatten(),
            layers.Dropout(1 - self.keep_prob),
            #
            #layers.BatchNormalization(),
            #
            layers.Dense(64, kernel_regularizer=regularizers.L1(0.001)),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dense(self.n_ligands)
        ]
    
    def call(self, x):
        ret = self.myConvLayer(x)
        for l in self.myLayers:
            ret = l(ret)
        return ret

class CovarLayer(layers.Layer):
    def __init__(self):
        super(CovarLayer, self).__init__()
    def call(self, x):
        ret = tf.matmul(tf.transpose(x, perm=[0, 2, 1]), x)
        scale = tf.cast(tf.shape(x)[1], tf.float32)
        return ret/scale

class ConvLayer(layers.Layer):
    def __init__(self,
        max_rho,
        n_ligands,
        n_thetas,
        n_rhos,
        n_rotations,
        feat_mask):
        
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
        
        
        initial_coords = self.compute_initial_coordinates()
        # self.rotation_angles = tf.Variable(np.arange(0, 2*np.pi, 2*np.pi/self.n_rotations).astype('float32'))
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
        
        for i in range(self.n_feat):
            self.mu_rho.append(
                self.add_weight(name="mu_rho_{}".format(i), shape=tf.shape(mu_rho_initial),
                                initializer = ValueInit(mu_rho_initial), trainable = True)
            )  # 1, n_gauss
            self.mu_theta.append(
                self.add_weight(name="mu_theta_{}".format(i), shape=tf.shape(mu_theta_initial),
                                initializer = ValueInit(mu_theta_initial), trainable = True)
            )  # 1, n_gauss
            self.sigma_rho.append(
                self.add_weight(name="sigma_rho_{}".format(i), shape=tf.shape(mu_rho_initial),
                                initializer = initializers.Constant(self.sigma_rho_init), trainable = True)
            )  # 1, n_gauss
            self.sigma_theta.append(
                self.add_weight(name="sigma_theta_{}".format(i), shape=tf.shape(mu_theta_initial),
                                initializer = initializers.Constant(self.sigma_theta_init), trainable = True)
            )  # 1, n_gauss
            
            self.b_conv.append(
                self.add_weight(
                    "b_conv_{}".format(i),
                    shape=[self.n_thetas * self.n_rhos], initializer='zeros',
                    trainable = True
                )
            )
            self.W_conv.append(
                self.add_weight(
                    "W_conv_{}".format(i),
                    shape=[self.n_thetas * self.n_rhos, self.n_thetas * self.n_rhos],
                    initializer=initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                    trainable = True
                )
            )
    
    def map_func(self, row):
        n_pockets = tf.cast(tf.shape(row)[0]/(8*200), dtype = tf.int32)
        bigShape = [n_pockets, 200, self.n_feat]
        smallShape = [n_pockets, 200, 1]
        idx = tf.cast(functools.reduce(prodFunc, bigShape), dtype = tf.int32)
        input_feat = tf.reshape(row[:idx], bigShape)
        rest = tf.reshape(row[idx:], [3] + smallShape)
        sample = tf.random.shuffle(tf.range(n_pockets))[:minPockets]
        data_list = [makeRagged(tsr) for tsr in [input_feat, rest[0], rest[1], rest[2]]]
        return [data_list, sample]
    
    def unpack_x(self, x):
        data_list, sample = tf.map_fn(fn=self.map_func, elems = x, fn_output_signature = [[inputFeatSpec, restSpec, restSpec, restSpec], sampleSpec])
        return [tf.gather(params = data, indices = sample, axis = 1, batch_dims = 1).to_tensor() for data in data_list]
    
    def call(self, x):
        input_feat, rho_coords, theta_coords, mask = self.unpack_x(x)
        
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

        return tf.stack(ret, axis=2)
    
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
        batches = tf.shape(input_feat)[0]
        
        n_samples = tf.shape(input=rho_coords)[1]
        n_vertices = tf.shape(input=rho_coords)[2]

        all_conv_feat = []
        for k in range(self.n_rotations):
            
            rho_coords_ = tf.reshape(rho_coords, [batches, -1, 1])  # batch_size*n_vertices
            thetas_coords_ = tf.reshape(theta_coords, [batches, -1, 1])  # batch_size*n_vertices

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
                gauss_activations, [batches, n_samples, n_vertices, -1]
            )  # batch_size, n_vertices, n_gauss
            gauss_activations = tf.multiply(gauss_activations, mask)
            if (
                mean_gauss_activation
            ):  # computes mean weights for the different gaussians
                
                gauss_activations /= (
                    tf.reduce_sum(input_tensor=gauss_activations, axis=2, keepdims=True) + eps
                )  # batch_size, n_vertices, n_gauss

            gauss_activations = tf.expand_dims(
                gauss_activations, 3
            )  # batch_size, n_vertices, 1, n_gauss,
            input_feat_ = tf.expand_dims(
                input_feat, 4
            )  # batch_size, n_vertices, n_feat, 1

            
            gauss_desc = tf.multiply(
                gauss_activations, input_feat_
            )  # batch_size, n_vertices, n_feat, n_gauss,
            gauss_desc = tf.reduce_sum(input_tensor=gauss_desc, axis=2)  # batch_size, n_feat, n_gauss,
            gauss_desc = tf.reshape(
                gauss_desc, [batches, n_samples, self.n_thetas * self.n_rhos]
            )  # batch_size, 80

            conv_feat = tf.matmul(gauss_desc, W_conv) + b_conv  # batch_size, 80
            all_conv_feat.append(conv_feat)
        all_conv_feat = tf.stack(all_conv_feat)
        conv_feat = tf.reduce_max(input_tensor=all_conv_feat, axis=0)
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
