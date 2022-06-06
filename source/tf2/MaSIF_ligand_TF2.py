import tensorflow as tf
import tf_slim as slim
import numpy as np
from tensorflow.keras import layers, Sequential, initializers, Model

#tf.debugging.set_log_device_placement(True)
npockets = 32

class MaSIF_ligand(Model):

    """
    The neural network model.
    """
    
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
    
    @tf.autograph.experimental.do_not_convert
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
            if (
                mean_gauss_activation
            ):  # computes mean weights for the different gaussians
                gauss_activations /= (
                    tf.reduce_sum(input_tensor=gauss_activations, axis=1, keepdims=True) + eps
                )  # batch_size, n_vertices, n_gauss

            gauss_activations = tf.expand_dims(
                gauss_activations, 2
            )  # batch_size, n_vertices, 1, n_gauss,
            input_feat_ = tf.expand_dims(
                input_feat, 3
            )  # batch_size, n_vertices, n_feat, 1

            gauss_desc = tf.multiply(
                gauss_activations, input_feat_
            )  # batch_size, n_vertices, n_feat, n_gauss,
            gauss_desc = tf.reduce_sum(input_tensor=gauss_desc, axis=1)  # batch_size, n_feat, n_gauss,
            gauss_desc = tf.reshape(
                gauss_desc, [n_samples, self.n_thetas * self.n_rhos]
            )  # batch_size, 80

            conv_feat = tf.matmul(gauss_desc, W_conv) + b_conv  # batch_size, 80
            all_conv_feat.append(conv_feat)
        all_conv_feat = tf.stack(all_conv_feat)
        conv_feat = tf.reduce_max(input_tensor=all_conv_feat, axis=0)
        conv_feat = tf.nn.relu(conv_feat)
        return conv_feat

    @tf.autograph.experimental.do_not_convert
    def bigPrepData(self, x):
        rho_coords = x['rho_coords']
        theta_coords = x['theta_coords']
        input_feat = x['input_feat']
        mask = x['mask']
        labels = x['labels']

        self.global_desc_1 = []
        b_conv = []
        for i in range(self.n_feat):
            b_conv.append(
                tf.Variable(
                    tf.zeros([self.n_thetas * self.n_rhos]),
                    name="b_conv_{}".format(i),
                )
            )
        for i in range(self.n_feat):
            my_input_feat = tf.expand_dims(input_feat[:, :, i], 2)

            #W_conv = tf.compat.v1.get_variable("W_conv_{}".format(i), shape=[self.n_thetas * self.n_rhos, self.n_thetas * self.n_rhos], initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
            W_conv = tf.Variable(
                initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")(shape=[
                    self.n_thetas * self.n_rhos,
                    self.n_thetas * self.n_rhos,
                ]),
                name = "W_conv_{}".format(i)
            )

            self.global_desc_1.append(
                self.inference(
                    my_input_feat,
                    rho_coords,
                    theta_coords,
                    mask,
                    W_conv,
                    b_conv[i],
                    self.mu_rho[i],
                    self.sigma_rho[i],
                    self.mu_theta[i],
                    self.sigma_theta[i],
                )
            )  # batch_size, n_gauss*1

        return tf.stack(self.global_desc_1, axis=1)
        
    def __init__(
        self,
        max_rho,
        n_ligands,
        n_thetas=16,
        n_rhos=5,
        n_gamma=1.0,
        learning_rate=1e-4,
        n_rotations=16,
        idx_gpu="/gpu:0",
        feat_mask=[1.0, 1.0, 1.0, 1.0],
        costfun="dprime",
        keep_prob = 1.0
    ):
        ## Call super - model initializer
        super().__init__()

        self.keep_prob = keep_prob
        
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

        tf.random.set_seed(0)
        
        
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
        for i in range(self.n_feat):
            self.mu_rho.append(
                tf.Variable(mu_rho_initial, name="mu_rho_{}".format(i))
            )  # 1, n_gauss
            self.mu_theta.append(
                tf.Variable(mu_theta_initial, name="mu_theta_{}".format(i))
            )  # 1, n_gauss
            self.sigma_rho.append(
                tf.Variable(
                    np.ones_like(mu_rho_initial) * self.sigma_rho_init,
                    name="sigma_rho_{}".format(i),
                )
            )  # 1, n_gauss
            self.sigma_theta.append(
                tf.Variable(
                    (np.ones_like(mu_theta_initial) * self.sigma_theta_init),
                    name="sigma_theta_{}".format(i),
                )
            )  # 1, n_gauss
        
        self.opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
 

        self.myLayers=[
            layers.Reshape([-1, self.n_thetas * self.n_rhos * self.n_feat], input_shape = [32, self.n_feat, self.n_thetas * self.n_rhos]),
            #layers.Lambda(lambda x : tf.reshape(x, [npockets, self.n_thetas * self.n_rhos * self.n_feat]), input_shape = [32, self.n_feat, self.n_thetas * self.n_rhos]),
            layers.Dense(self.n_thetas * self.n_rhos, activation="relu"),
            layers.Lambda(self.lambdaLayer),
            #layers.Reshape([1, -1]),
            layers.Flatten(),
            layers.Dropout(1 - self.keep_prob),
            layers.Dense(64, activation="relu"),
            layers.Dense(self.n_ligands, activation="relu")
        ]

    
    def lambdaLayer(self, x):
        x = tf.squeeze(x)
        numer = tf.matmul(tf.transpose(x), x)
        denom = tf.cast(tf.shape(x)[0], tf.float32)
        return tf.expand_dims(numer/denom, axis=0)

    
    def call(self, inputs):
        ret = inputs
        for l in self.myLayers:
            ret = l(ret)
        return ret
    
    
