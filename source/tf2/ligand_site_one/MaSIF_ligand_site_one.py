import numpy as np
from tensorflow.keras import layers, Sequential, initializers, Model
import tensorflow as tf
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
        n_conv_layers = 1,
        conv_batch_size = 100
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
 
        self.myConvLayer = ConvLayer(max_rho, n_ligands, n_thetas, n_rhos, n_rotations, feat_mask, n_conv_layers, conv_batch_size)
        self.myDense = layers.Dense(self.n_thetas, activation="relu")
        self.outLayer = layers.Dense(1)
        
        '''
        self.myLayers=[
            myConvLayer, 
            #layers.Dense(self.n_thetas * self.n_rhos, activation="relu"),
            #layers.Dense(30, activation='relu'),
            #layers.Dense(10, activation='relu'),
            layers.Dense(1)
        ]'''
    '''
    def train_step(self, data):
        print('Train step')
        
        x, y = data
        
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)
        
        return {m.name: m.result() for m in self.metrics}'''
    
    
    def call(self, x, training=False):
        tf.config.set_soft_device_placement(True)
        ret = self.myConvLayer(x)
        ret = self.myDense(ret)
        ret = self.outLayer(ret)
        return ret

class ConvLayer(layers.Layer):
    def __init__(self,
        max_rho,
        n_ligands,
        n_thetas,
        n_rhos,
        n_rotations,
        feat_mask,
        n_conv_layers,
        conv_batch_size):
        
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
        
        self.conv_batch_size = conv_batch_size
        
        # Variable dict lists
        self.variable_dicts = []
        
        initial_coords = self.compute_initial_coordinates()
        # self.rotation_angles = tf.Variable(np.arange(0, 2*np.pi, 2*np.pi/self.n_rotations).astype('float32'))
        
        self.conv_shapes = [[self.n_thetas * self.n_rhos, self.n_thetas * self.n_rhos],
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
        
        mu_rho_initial = tf.cast(tf.expand_dims(initial_coords[:, 0], axis=0), dtype=tf.float32)
        mu_theta_initial = tf.cast(tf.expand_dims(initial_coords[:, 1], axis=0), dtype=tf.float32)
        
        mu_rho = []
        mu_theta = []
        sigma_rho = []
        sigma_theta = []

        b_conv = []
        W_conv = []
        
        layer_num = 0
        for i in range(self.n_feat):
            mu_rho.append(
                self.add_weight(name="mu_rho_{}_{}".format(i, layer_num), shape=tf.shape(mu_rho_initial),
                                initializer = ValueInit(mu_rho_initial), trainable = True)
            )  # 1, n_gauss
            mu_theta.append(
                self.add_weight(name="mu_theta_{}_{}".format(i, layer_num), shape=tf.shape(mu_theta_initial),
                                initializer = ValueInit(mu_theta_initial), trainable = True)
            )  # 1, n_gauss
            sigma_rho.append(
                self.add_weight(name="sigma_rho_{}_{}".format(i, layer_num), shape=tf.shape(mu_rho_initial),
                                initializer = initializers.Constant(self.sigma_rho_init), trainable = True)
            )  # 1, n_gauss
            sigma_theta.append(
                self.add_weight(name="sigma_theta_{}_{}".format(i, layer_num), shape=tf.shape(mu_theta_initial),
                                initializer = initializers.Constant(self.sigma_theta_init), trainable = True)
            )  # 1, n_gauss


            b_conv.append(
                self.add_weight(
                    "b_conv_{}_{}".format(i, layer_num),
                    shape=self.conv_shapes[layer_num][1], initializer='zeros',
                    trainable = True
                )
            )
            W_conv.append(
                self.add_weight(
                    "W_conv_{}_{}".format(i, layer_num),
                    shape=self.conv_shapes[layer_num], initializer=initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
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
        
        gu_init = tf.keras.initializers.GlorotUniform()
        '''FC1_W = self.add_weight('FC1_W', shape=(self.n_thetas * self.n_rhos * self.n_feat, 200), initializer = gu_init, trainable=True, dtype="float32")
        FC1_b = self.add_weight('FC1_b', shape=(200,), initializer = zero_init, trainable=True, dtype="float32")'''
        var_dict['FC1_W'] = self.add_weight('FC1_W', shape=(self.n_thetas * self.n_rhos * self.n_feat, self.n_thetas * self.n_rhos), initializer = gu_init, trainable=True, dtype="float32")
        var_dict['FC1_b'] = self.add_weight('FC1_b', shape=(self.n_thetas * self.n_rhos,), initializer = 'zeros', trainable=True, dtype="float32")

        var_dict['FC2_W'] = self.add_weight('FC2_W', shape=(self.n_thetas * self.n_rhos, self.n_feat), initializer = gu_init, trainable=True, dtype="float32")
        var_dict['FC2_b'] = self.add_weight('FC2_b', shape=(self.n_feat,), initializer = 'zeros', trainable=True, dtype="float32")
        
        self.variable_dicts.append(var_dict)
        
        i = 0
        for layer_num in range(1, n_conv_layers):
            var_dict = {}
            var_dict['mu_rho'] = self.add_weight(name="mu_rho_{}_{}".format(i, layer_num), shape=tf.shape(mu_rho_initial), initializer = ValueInit(mu_rho_initial), trainable = True)
            var_dict['mu_theta'] = self.add_weight(name="mu_theta_{}_{}".format(i, layer_num), shape=tf.shape(mu_theta_initial), initializer = ValueInit(mu_theta_initial), trainable = True)
            var_dict['sigma_rho'] = self.add_weight(name="sigma_rho_{}_{}".format(i, layer_num), shape=tf.shape(mu_rho_initial), initializer = initializers.Constant(self.sigma_rho_init), trainable = True)
            var_dict['sigma_theta'] = self.add_weight(name="sigma_theta_{}_{}".format(i, layer_num), shape=tf.shape(mu_theta_initial), initializer = initializers.Constant(self.sigma_theta_init), trainable = True)
            
            var_dict['b_conv'] = self.add_weight("b_conv_{}_{}".format(i, layer_num), shape=self.conv_shapes[layer_num][1], initializer='zeros', trainable = True)
            var_dict['W_conv'] = self.add_weight("W_conv_{}_{}".format(i, layer_num), shape=self.conv_shapes[layer_num], initializer=initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), trainable = True)
            
            self.variable_dicts.append(var_dict)
    '''
    #@tf.autograph.experimental.do_not_convert
    def map_func(self, row, makeSample):
        input_feat = tf.reshape(row[:self.bigIdx], self.bigShape)
        rest = tf.reshape(row[self.bigIdx:], [3] + self.smallShape)
        return [input_feat, rest[0], rest[1], rest[2]]
    '''
    
    def call(self, x):
        '''input_feat, rho_coords, theta_coords, mask = tf.map_fn(fn=self.map_func, elems = x,
                              fn_output_signature = [inputFeatSpec, restSpec, restSpec, restSpec])'''

        '''input_feat = tf.cast(tf.gather(x, tf.range(5), axis=-1), dtype=tf.float32)
        rho_coords = tf.cast(tf.gather(x, 5, axis=-1), dtype=tf.float32)
        theta_coords = tf.cast(tf.gather(x, 6, axis=-1), dtype=tf.float32)
        mask = tf.cast(tf.expand_dims(tf.gather(x, 7, axis=-1), axis=-1), dtype=tf.float32)
        ####
        indices_tensor = tf.cast(tf.gather(x, 8, axis=-1), dtype=tf.int32)
        ####'''
        
        print(f'Conv layer: 0')
        
        var_dict = self.variable_dicts[0]
        
        n_samples = tf.shape(x[1])[0]
        if self.conv_batch_size is None:
            input_feat, rho_coords, theta_coords, mask = [tf.cast(tsr, dtype=tf.float32) for tsr in x[0]]
            indices_tensor = tf.cast(x[1], dtype=tf.int32)
            sampIdx = tf.stack([0, n_samples], axis=0)
        else:
            print('here')
            leftover = self.conv_batch_size - (n_samples % self.conv_batch_size)
            def addLeftover(tsr, dtype):
                shape = tf.concat([tf.expand_dims(leftover, axis=0), tsr.shape[1:]], axis=0)
                empty = tf.zeros(shape, dtype=dtype)
                return tf.concat([tsr, empty], axis=0)
            input_feat, rho_coords, theta_coords, mask = (addLeftover(tsr, tf.float32) for tsr in x[0])
            indices_tensor = addLeftover(x[1], tf.int32)
            sampIdx = tf.range(n_samples + leftover + 1, delta=self.conv_batch_size)
        
        
        ret = []
        for i in range(self.n_feat):
            my_input_feat = tf.gather(input_feat, tf.range(i, i+1), axis=-1)
            
            def tempInference(idx):
                sample = tf.range(sampIdx[idx], sampIdx[idx+1])
                input_feat_temp = tf.gather(my_input_feat, sample, axis=0)
                rho_coords_temp = tf.gather(rho_coords, sample, axis=0)
                theta_coords_temp = tf.gather(theta_coords, sample, axis=0)
                mask_temp = tf.gather(mask, sample, axis=0)
                
                return self.inference(
                    input_feat_temp,
                    rho_coords_temp,
                    theta_coords_temp,
                    mask_temp,
                    var_dict['W_conv'][i],
                    var_dict['b_conv'][i],
                    var_dict['mu_rho'][i],
                    var_dict['sigma_rho'][i],
                    var_dict['mu_theta'][i],
                    var_dict['sigma_theta'][i]
                )
            
            map_output = tf.map_fn(fn=tempInference, elems = tf.range(tf.shape(sampIdx)[0]-1), fn_output_signature = tf.TensorSpec(shape=[self.conv_batch_size, self.conv_shapes[0][0]], dtype=tf.float32))
            #ret.append(tf.concat(tf.unstack(map_output), axis=0))
            ret.append(tf.reshape(map_output, shape=[-1, map_output.shape[-1]]))

        ret = tf.stack(ret, axis=2)
        ret = tf.reshape(ret, self.reshape_shapes[0])
        
        ret = tf.matmul(ret, var_dict['FC1_W']) + var_dict['FC1_b']
        ret = tf.nn.relu(ret)
        
        ret = tf.matmul(ret, var_dict['FC2_W']) + var_dict['FC2_b']
        ret = tf.nn.relu(ret)
        
        #####################
        if len(self.variable_dicts) == 1:
            return ret
        ####################
        
        start = 1
        for layer_num, var_dict in enumerate(self.variable_dicts[start:], start):
            print(f'Conv layer: {layer_num}')
            
            if layer_num == 0:
                continue

            input_feat = tf.gather(ret, indices_tensor)
            
            def tempInference(idx):
                sample = tf.range(sampIdx[idx], sampIdx[idx+1])
                input_feat_temp = tf.gather(input_feat, sample, axis=0)
                rho_coords_temp = tf.gather(rho_coords, sample, axis=0)
                theta_coords_temp = tf.gather(theta_coords, sample, axis=0)
                mask_temp = tf.gather(mask, sample, axis=0)
                
                return self.inference(
                    input_feat_temp,
                    rho_coords_temp,
                    theta_coords_temp,
                    mask_temp,
                    var_dict['W_conv'],
                    var_dict['b_conv'],
                    var_dict['mu_rho'],
                    var_dict['sigma_rho'],
                    var_dict['mu_theta'],
                    var_dict['sigma_theta']
                )
            
            map_output = tf.map_fn(fn=tempInference, elems = tf.range(tf.shape(sampIdx)[0]-1), fn_output_signature = tf.TensorSpec(shape=[self.conv_batch_size, self.conv_shapes[layer_num][0]], dtype=tf.float32))
            #ret = tf.concat(tf.unstack(map_output), axis=0)
            ret = tf.reshape(map_output, shape=[-1, map_output.shape[-1]])
            
            # Reduce the dimensionality by averaging over the last dimension
            ret = tf.reshape(ret, self.reshape_shapes[layer_num])
            ret = self.reduce_funcs[layer_num](ret)
                
        return ret[:tf.shape(ret)[0] - leftover]
        
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
                    tf.reduce_sum(input_tensor=gauss_activations, axis=2, keepdims=True) + eps
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
