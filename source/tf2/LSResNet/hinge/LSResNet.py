import numpy as np
from tensorflow.keras import layers, Sequential, initializers, Model, regularizers, losses, metrics
import tensorflow as tf
from tensorflow.keras import backend as K
import functools
from operator import add
from default_config.util import *

def runLayers(layers, x):
    for l in layers:
        x = l(x)
    return x

hinge_inst = losses.Hinge()
def myF1(y_true, y_pred, threshold=0.5):
    y_true = tf.reshape(y_true, [-1]) > 0.0
    y_pred = tf.reshape(y_pred, [-1]) > 0.0
    overlap = tf.reduce_sum(tf.cast(y_true & y_pred, dtype=tf.float32))
    n_true = tf.reduce_sum(tf.cast(y_true, dtype=tf.float32))
    n_pred = tf.reduce_sum(tf.cast(y_pred, dtype=tf.float32))
    recall = overlap/n_true
    precision = overlap/n_pred
    f1 = 2*precision*recall / (precision + recall)
    return tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)

class LSResNet(Model):
    def train_step(self, data):
        '''for m in self.metrics:
            m.reset_states()
            print('reset')'''
        
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = tf.pow(hinge_inst(y, y_pred), self.hinge_p)
            if not self.specialNeuron is None:
                loss += self.specialNeuron.reg_loss()

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        self.loss_tracker.update_state(loss)
        #for m in self.metrics[1:]:
        #    m.update_state(y, y_pred)
        #self.f1_metric.update_state(myF1(y, y_pred))
        self.f1_metric.update_state(y, y_pred)
        self.hinge_acc_metric.update_state(y, y_pred)
        
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        loss = tf.pow(hinge_inst(y, y_pred), self.hinge_p)
        if not self.specialNeuron is None:
            loss += self.specialNeuron.reg_loss()
            
        self.loss_tracker.update_state(loss)
        self.f1_metric.update_state(y, y_pred)
        self.hinge_acc_metric.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}
    
    @property
    def metrics(self):
        return [self.loss_tracker, self.f1_metric, self.hinge_acc_metric]
    
    def __init__(
        self,
        max_rho,
        n_thetas=16,
        n_rhos=5,
        learning_rate=1e-4,
        n_rotations=16,
        feat_mask=[1.0, 1.0, 1.0, 1.0],
        keep_prob = 1.0,
        reg_val = 0.0,
        reg_type = 'l2',
        hinge_p = 3,
        use_special_neuron = False,
        reg_const = 1e-2
    ):
        ## Call super - model initializer
        super(LSResNet, self).__init__()
        
        regKwargs = {reg_type : reg_val}
        self.reg = regularizers.L1L2(**regKwargs)
        
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
        
        self.scale = 0.5
        self.max_dist = 35
        
        self.opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        #self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits = True)
        
        self.hinge_p = hinge_p
        
        self.loss_tracker = metrics.Mean(name="loss")
        self.f1_metric = F1_Metric()
        #self.f1_metric = metrics.Mean(name='F1')
        self.hinge_acc_metric = HingeAccuracy()
        
        self.conv_shapes = [[self.n_thetas * self.n_rhos, self.n_thetas * self.n_rhos],
                       [self.n_feat * self.n_thetas * self.n_rhos, self.n_feat * self.n_thetas * self.n_rhos],
                       [self.n_feat * self.n_thetas * self.n_rhos, self.n_feat * self.n_thetas * self.n_rhos],
                       [self.n_thetas * self.n_rhos * self.n_thetas * self.n_rhos, self.n_thetas * self.n_rhos * self.n_thetas * self.n_rhos]] 
        self.reshape_shapes = [[-1, self.n_thetas * self.n_rhos * self.n_feat],
                          [-1, self.n_feat, self.n_thetas * self.n_rhos],
                          [-1, self.n_feat, self.n_thetas * self.n_rhos],
                          [-1, self.n_thetas * self.n_rhos, self.n_thetas * self.n_rhos]]


        self.convBlock0 = [
            ConvLayer(5, self.conv_shapes[0], self.max_rho, self.n_thetas, self.n_rhos, self.n_rotations, self.n_feat, self.reg),
            layers.Reshape(self.reshape_shapes[0])
        ]
        
        self.denseReduce = [
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dense(self.n_thetas * self.n_rhos, activation="relu"),
            layers.Dense(self.n_feat, activation="relu"),
        ]
        
        resolution = 1. / self.scale
        self.myMakeGrid = MakeGrid(max_dist=self.max_dist, grid_resolution=resolution)
        
        self.box_size = 36
        
        ####
        #self.flatten = layers.Flatten()
        if use_special_neuron:
            self.reshapeBefore = layers.Reshape((self.box_size**3, 5))
            self.specialNeuron = EXP_Neuron(input_dim=5, units=1, reg_const=reg_const)
            self.reshapeAfter = layers.Reshape((self.box_size, self.box_size, self.box_size, 1))
        else:
            self.specialNeuron = None
        ####
        
        if K.image_data_format()=='channels_last':
            bn_axis=4
        else:
            bn_axis=1
            
        f=5
        filters = [f, f, f]
        filters1,filters2,filters3=filters
        strides = (1,1,1)
        
        self.RNConvBlock=[
            [layers.Conv3D(filters1, kernel_size=1, strides=strides, data_format=K.image_data_format()),
            layers.BatchNormalization(axis=bn_axis),
            layers.ReLU(),
             
            layers.Conv3D(filters2, kernel_size=3, padding='same'),
            layers.BatchNormalization(axis=bn_axis),
            layers.ReLU(),
             
            layers.Conv3D(filters3, kernel_size=1),
            layers.BatchNormalization(axis=bn_axis)],
            
            [layers.Conv3D(filters3, kernel_size=1, strides=strides),
            layers.BatchNormalization(axis=bn_axis)]
        ]
        
        self.lastConvLayer = layers.Conv3D(1, kernel_size=1, activation='tanh')
        
    def call(self, X_packed, training=False):
        X, xyz_coords = X_packed
        
        ret = runLayers(self.convBlock0, X)
        ret = runLayers(self.denseReduce, ret)
        
        ret = self.myMakeGrid(xyz_coords, ret)
        
        #####
        if not self.specialNeuron is None:
            #flatRet = self.flatten(ret)
            #flatRet = tf.reshape(ret, (self.box_size**3, -1))
            flatRet = self.reshapeBefore(ret)
            expOutput = self.specialNeuron(flatRet)
            expOutput = self.reshapeAfter(expOutput)
            #expOutput = tf.reshape(expOutput, (self.box_size, self.box_size, self.box_size, 1))
        #####
        
        ret1 = runLayers(self.RNConvBlock[0], ret)
        residue = runLayers(self.RNConvBlock[1], ret)
        ret = tf.add(ret1, residue)
        
        ret = tf.nn.relu(ret)
        ret = self.lastConvLayer(ret)
        
        #####
        if not self.specialNeuron is None:
            ret = tf.add(ret, expOutput)
        #####
        
        return ret

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
                                initializer = ValueInit(mu_rho_initial), trainable = True, regularizer=reg)
            )  # 1, n_gauss
            self.mu_theta.append(
                self.add_weight("mu_theta_{}".format(i), shape=tf.shape(mu_theta_initial),
                                initializer = ValueInit(mu_theta_initial), trainable = True, regularizer=reg)
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

class EXP_Neuron(layers.Layer):
    def __init__(self, input_dim, units, reg_const):
        super(EXP_Neuron, self).__init__()
        self.reg_const = reg_const
        self.a = self.add_weight('EXP_a', initializer="random_normal", trainable=True)
        self.w = self.add_weight('EXP_w',
            shape=(input_dim, units), initializer="random_normal", trainable=True
        )
        self.b = self.add_weight('EXP_b', shape=(units), initializer="zeros", trainable=True)
    def call(self, inputs):
        return self.a * tf.exp(tf.matmul(inputs, self.w) + self.b)
    def reg_loss(self):
        return self.reg_const * tf.square(self.a) / 2

class HingeAccuracy(metrics.Metric):
    def __init__(self, name='hinge_accuracy', **kwargs):
        super(HingeAccuracy, self).__init__(name=name, **kwargs)
        self.hinge_acc_score = self.add_weight(name='hinge_acc_score', initializer='zeros')
        self.n_matching = self.add_weight(name='n_matching', initializer='zeros')
        self.n_total = self.add_weight(name='n_total', initializer='zeros')
    def update_state(self, y_true, y_pred):
        y_true = tf.reshape(y_true, [-1]) > 0.0
        y_pred = tf.reshape(y_pred, [-1]) > 0.0
        result = tf.cast(y_true == y_pred, tf.float32)
        n_matching = tf.reduce_sum(result)
        n_total = tf.shape(result)[0]
        
        self.n_matching.assign_add(tf.cast(n_matching, self.dtype))
        self.n_total.assign_add(tf.cast(n_total, self.dtype))
    def result(self):
        self.hinge_acc_score = self.n_matching/self.n_total
        return self.hinge_acc_score

class F1_Metric(metrics.Metric):
    def __init__(self, name='F1', **kwargs):
        super(F1_Metric, self).__init__(name=name, **kwargs)
        self.f1_score = self.add_weight(name='f1_score', initializer='zeros')
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
    def update_state(self, y_true, y_pred):
        y_true = tf.reshape(y_true, [-1]) > 0.0
        y_pred = tf.reshape(y_pred, [-1]) > 0.0
        overlap = tf.reduce_sum(tf.cast(y_true & y_pred, dtype=tf.float32))
        n_true = tf.reduce_sum(tf.cast(y_true, dtype=tf.float32))
        n_pred = tf.reduce_sum(tf.cast(y_pred, dtype=tf.float32))
        recall = overlap/n_true
        precision = overlap/n_pred
        f1 = 2*precision*recall / (precision + recall)
        f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
        self.total.assign_add(tf.cast(f1, self.dtype))
        self.count.assign_add(tf.cast(1, self.dtype))
    def result(self):
        return self.total/self.count
