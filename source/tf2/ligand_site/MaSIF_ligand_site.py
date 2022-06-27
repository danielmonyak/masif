import numpy as np
from tensorflow.keras import layers, Sequential, initializers, Model
import functools
from default_config.util import *

params = masif_opts["ligand"]
minPockets = params['minPockets']
prepSize = 2 * params['savedPockets']

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
        #self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits = False)
        self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits = True)
 
        
        self.myConvLayer = ConvLayer(max_rho, n_ligands, n_thetas, n_rhos, n_rotations, feat_mask, n_conv_layers)
        
        self.myLayers=[
            #layers.Reshape([minPockets, self.n_feat * self.n_thetas * self.n_rhos]),
            
            layers.Dense(self.n_thetas * self.n_rhos, activation="relu"),
            
            layers.Dropout(1 - self.keep_prob),
            #layers.Dense(64, activation="relu"),
            layers.Dense(30, activation='relu'),
            #layers.Dense(10, activation='relu'),
            layers.Dense(1)
        ]
        '''self.myLayers=[
            layers.Dense(self.n_thetas, activation="relu"),
            layers.Dense(1)
        ]'''
        
    
    def map_func(self, row):
        n_pockets = tf.shape(row)[0]
        sample = tf.random.shuffle(tf.range(n_pockets))[:minPockets]
        return sample
    def make_y(self, y_raw):
        sample = tf.map_fn(fn=self.map_func, elems = y_raw, fn_output_signature = sampleSpec)
        return (tf.gather(params = y_raw, indices = sample, axis = 1, batch_dims = 1), sample)
    
    def train_step(self, data):
        if len(data) == 3:
            x, y_raw, class_weight = data
        else:
            x, y_raw = data
            class_weight = None
        
        y, sample = self.make_y(y_raw)
        
        with tf.GradientTape() as tape:
            y_pred = self(x, sample = sample, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        if len(data) == 3:
            x, y_raw, class_weight = data
        else:
            x, y_raw = data
            class_weight = None
        
        y, sample = self.make_y(y_raw)
        
        y_pred = self(x, sample = sample, training=False)
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}
    
    def call(self, x, sample = None):
        #if sample is None:
        #    sample = tf.zeros([1, minPockets])
        ret = self.myConvLayer(x, sample)
        for l in self.myLayers:
            ret = l(ret)
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
        
        
        # Variable dict lists
        self.variable_dicts = []
        self.relu = tf.keras.layers.ReLU()
        self.Map_func = lambda row : self.map_func(row)
        self.Map_func_sample = lambda row : self.map_func(row, makeSample = True)
        
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
            
        
        #gu_init = tf.keras.initializers.GlorotUniform()
        #zero_init = tf.keras.initializers.Zeros
        #FC1_W = self.add_weight('FC1_W', shape=(self.n_thetas * self.n_rhos * self.n_feat, 200), initializer = gu_init, trainable=True, dtype="float32")
        #FC1_b = self.add_weight('FC1_b', shape=(200,), initializer = zero_init, trainable=True, dtype="float32")
        '''FC1_W = self.add_weight('FC1_W', shape=(self.n_thetas * self.n_rhos * self.n_feat, self.n_thetas * self.n_rhos), initializer = gu_init, trainable=True, dtype="float32")
        FC1_b = self.add_weight('FC1_b', shape=(self.n_thetas * self.n_rhos,), initializer = zero_init, trainable=True, dtype="float32")
        
        FC2_W = self.add_weight('FC2_W', shape=(self.n_thetas * self.n_rhos, self.n_feat), initializer = gu_init, trainable=True, dtype="float32")
        FC2_b = self.add_weight('FC2_b', shape=(self.n_feat,), initializer = zero_init, trainable=True, dtype="float32")'''
        
        var_dict = {}
        var_dict['mu_rho'] = mu_rho
        var_dict['mu_theta'] = mu_theta
        var_dict['sigma_rho'] = sigma_rho
        var_dict['sigma_theta'] = sigma_theta
        var_dict['b_conv'] = b_conv
        var_dict['W_conv'] = W_conv
        '''
        var_dict['FC1_W'] = FC1_W
        var_dict['FC1_b'] = FC1_b
        
        var_dict['FC2_W'] = FC2_W
        var_dict['FC2_b'] = FC2_b'''
        
        self.variable_dicts.append(var_dict)
        
        
        i = 0
        for layer_num in range(1, n_conv_layers):
            var_dict = {}
            var_dict['mu_rho'] = tf.Variable(mu_rho_initial, name="mu_rho_{}_{}".format(i, layer_num), trainable = True)
            var_dict['mu_theta'] = tf.Variable(mu_theta_initial, name="mu_theta_{}_{}".format(i, layer_num), trainable = True)
            var_dict['sigma_rho'] = tf.Variable(np.ones_like(mu_rho_initial) * self.sigma_rho_init, name="sigma_rho_{}_{}".format(i, layer_num), trainable = True)
            var_dict['sigma_theta'] = tf.Variable((np.ones_like(mu_theta_initial) * self.sigma_theta_init), name="sigma_theta_{}_{}".format(i, layer_num), trainable = True)
            
            var_dict['b_conv'] = self.add_weight("b_conv_{}_{}".format(i, layer_num), shape=conv_shapes[layer_num][1], initializer='zeros', trainable = True)
            var_dict['W_conv'] = self.add_weight("W_conv_{}_{}".format(i, layer_num), shape=conv_shapes[layer_num], initializer=initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),, trainable = True)
            '''var_dict['b_conv'] = tf.Variable(tf.zeros(conv_shapes[0][1]), name="b_conv_{}_{}".format(i, layer_num), trainable = True)
            var_dict['W_conv'] = tf.Variable(initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")(shape=conv_shapes[0]),
                                             name = "W_conv_{}_{}".format(i, layer_num), trainable = True)'''
            
            #################################
            '''var_dict['FC1_W'] = self.add_weight('FC1_W', shape=(self.n_thetas * self.n_rhos, 200), initializer = gu_init, trainable=True, dtype="float32")
            var_dict['FC1_b'] = self.add_weight('FC1_b', shape=(200,), initializer = zero_init, trainable=True, dtype="float32")'''
            #################################
            self.variable_dicts.append(var_dict)
    
    def map_func(self, row, makeSample = False):
        n_pockets = tf.cast(tf.shape(row)[0]/(8*200), dtype = tf.int32)
        bigShape = [n_pockets, 200, self.n_feat]
        smallShape = [n_pockets, 200, 1]
        idx = tf.cast(functools.reduce(prodFunc, bigShape), dtype = tf.int32)
        input_feat = tf.reshape(row[:idx], bigShape)
        rest = tf.reshape(row[idx:], [3] + smallShape)
        data_list = [makeRagged(tsr) for tsr in [input_feat, rest[0], rest[1], rest[2]]]
        if not makeSample:
            return data_list
        sample = tf.random.shuffle(tf.range(n_pockets))[:minPockets]
        return [data_list, sample]

    def unpack_x(self, x, sample):
        if sample is None:
            data_list, sample = tf.map_fn(fn=self.Map_func_sample, elems = x,
                                              fn_output_signature = [[inputFeatSpec, restSpec, restSpec, restSpec], sampleSpec])
        else:
            data_list = tf.map_fn(fn=self.Map_func, elems = x,
                                          fn_output_signature = [inputFeatSpec, restSpec, restSpec, restSpec])
        return [tf.gather(params = data, indices = sample, axis = 1, batch_dims = 1).to_tensor() for data in data_list]
    
    #@tf.function(input_signature=[
    #                tf.RaggedTensorSpec(tf.TensorShape([None, None]), tf.float32, 1, tf.int64),
    #                tf.TensorSpec(shape=tf.TensorShape([None, minPockets]), dtype=tf.int32, name=None)
    #])
    def call(self, x, sample):
        input_feat, rho_coords, theta_coords, mask = self.unpack_x(x, sample)

        var_dict = self.variable_dicts[0]

        mu_rho = var_dict['mu_rho']
        mu_theta = var_dict['mu_theta']
        sigma_rho = var_dict['sigma_rho']
        sigma_theta = var_dict['sigma_theta']
        b_conv = var_dict['b_conv']
        W_conv = var_dict['W_conv']
        '''
        FC1_W = var_dict['FC1_W']
        FC1_b = var_dict['FC1_b']
        
        FC2_W = var_dict['FC2_W']
        FC2_b = var_dict['FC2_b']'''


        ret = []
        for i in range(self.n_feat):
            my_input_feat = tf.gather(input_feat, tf.range(i, i+1), axis=-1)
            ret.append(
                self.inference(
                    my_input_feat,
                    rho_coords,
                    theta_coords,
                    mask,
                    W_conv[i],
                    b_conv[i],
                    mu_rho[i],
                    sigma_rho[i],
                    mu_theta[i],
                    sigma_theta[i],
                )
            )  # batch_size, n_gauss*1

        ret = tf.stack(ret, axis=2)
        ret = tf.reshape(ret, self.reshape_shapes[0])
        '''
        ret = tf.matmul(ret, FC1_W) + FC1_b
        ret = self.relu(ret)
        
        ret = tf.matmul(ret, FC2_W) + FC2_b
        ret = self.relu(ret)
        '''
        #####################
        if len(self.variable_dicts) == 1:
            return ret
        ####################
        for layer_num, var_dict in enumerate(self.variable_dicts):
            print('SHOULD NOT BE HERE')
            
            if layer_num == 0:
                continue

            mu_rho = var_dict['mu_rho']
            mu_theta = var_dict['mu_theta']
            sigma_rho = var_dict['sigma_rho']
            sigma_theta = var_dict['sigma_theta']

            b_conv = var_dict['b_conv']
            W_conv = var_dict['W_conv']

            FC1_W = var_dict['FC1_W']
            FC1_b = var_dict['FC1_b']

            ret = self.inference(
                tf.expand_dims(ret, axis=-1),
                rho_coords,
                theta_coords,
                mask,
                W_conv,
                b_conv,
                mu_rho,
                sigma_rho,
                mu_theta,
                sigma_theta,
            )  # batch_size, n_gauss*n_gauss
            # Reduce the dimensionality by averaging over the last dimension
            '''ret = tf.reshape(ret, self.reshape_shapes[layer_num])
            ret = self.reduce_funcs[layer_num](ret)'''
            
            #ret = tf.matmul(ret, FC1_W) + FC1_b
            ret = self.relu(ret)

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
        batches = tf.shape(input_feat)[0]

        n_samples = tf.shape(input=rho_coords)[1]
        n_vertices = tf.shape(input=rho_coords)[2]

        #n_feat = tf.shape(input_feat)[2]

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

            if mean_gauss_activation:  # computes mean weights for the different gaussians
                gauss_activations /= (
                    tf.reduce_sum(input_tensor=gauss_activations, axis=2, keepdims=True) + eps
                )  # batch_size, n_vertices, n_gauss

            gauss_activations = tf.expand_dims(
                gauss_activations, 3
            )  # batch_size, n_vertices, 1, n_gauss,

            # check the axis on this
            input_feat_ = tf.expand_dims(
                input_feat, -1
            )  # batch_size, n_vertices, n_feat, 1


            gauss_desc = tf.multiply(
                gauss_activations, input_feat_
            )  # batch_size, n_vertices, n_feat, n_gauss,
            gauss_desc = tf.reduce_sum(gauss_desc, axis=2)  # batch_size, n_feat, n_gauss,
            gauss_desc = tf.reshape(
                gauss_desc, [batches, n_samples, self.n_thetas * self.n_rhos]
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
