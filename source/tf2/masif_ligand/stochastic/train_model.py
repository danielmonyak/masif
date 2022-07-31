import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import sys
import numpy as np
import tensorflow as tf

phys_gpus = tf.config.list_physical_devices('GPU')
for phys_g in phys_gpus:
    tf.config.experimental.set_memory_growth(phys_g, True)

import default_config.util as util
from default_config.masif_opts import masif_opts
from tf2.masif_ligand.stochastic.MaSIF_ligand import MaSIF_ligand
from tf2.masif_ligand.stochastic.get_data import get_data

#lr = 1e-3
# Try this learning rate after

reg_val = 0.0
reg_type = 'l2'

continue_training = False
dev = '/GPU:3'
cpu = '/CPU:0'

params = masif_opts["ligand"]
defaultCode = params['defaultCode']

training = np.load('')
val = np.load('')

train_iter = iter(training)
val_iter = iter(val)

modelDir = 'kerasModel'
ckpPath = os.path.join(modelDir, 'ckp')
modelPath = os.path.join(modelDir, 'savedModel')

model = MaSIF_ligand(
    params["max_distance"],
    train_y.shape[1],
    feat_mask=params["feat_mask"],
    reg_val = reg_val, reg_type = reg_type,
    keep_prob=1.0
)
model.compile(optimizer = model.opt,
    loss = model.loss_fn,
    metrics = ['categorical_accuracy']
)  
if continue_training:
    model.load_weights(ckpPath)
    last_epoch = 18
    initValThresh = 0.71429
else:
    last_epoch = 0
    initValThresh = 0

saveCheckpoints = tf.keras.callbacks.ModelCheckpoint(
    ckpPath,
    monitor = 'val_categorical_accuracy',
    #save_best_only = True,
    verbose = 1,
    initial_value_threshold = initValThresh
)


optimizer = keras.optimizers.SGD(learning_rate=1e-3)
loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)

train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
val_acc_metric = keras.metrics.CategoricalAccuracy()

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_acc_metric.update_state(y, logits)
    return loss_value

iterations = 0
n_train = 100
while iterations < 1e4:
    for i in range(n_train):
        try:
            pdb_id = next(train_iter)
        except:
            train_iter = iter(training)
        
        X, pocket_points = get_data(pdb_id, training=True)
        n_samples = X[0].shape[1]
        y = np.empty([1, n_samples, 1], dtype=np.int32)
        for pp in pocket_points:
            X_temp = tuple(arr[:, pp] for arr in X)
            y.fill(0)
            y[0, pp, 0] = 1
            loss_value = train_step(X_temp, y)
        
        
model.save(modelPath)
