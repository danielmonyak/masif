# Header variables and parameters.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 


import numpy as np
from IPython.core.debugger import set_trace
import importlib
import sys
from default_config.masif_opts import masif_opts
from tf2.read_ligand_tfrecords import _parse_function
import tensorflow as tf

params = masif_opts["ligand"]
defaultCode = params['defaultCode']
n_classes = params['n_classes']

outdir = '/data02/daniel/masif/datasets/ligand_site'
genOutPath = os.path.join(outdir, '{}_{}.npy')

def helper(feed_dict):
    def helperInner(tsr_key):
        tsr = feed_dict[tsr_key]
        return tf.reshape(tsr, [-1])
    key_list = ['input_feat', 'rho_coords', 'theta_coords', 'mask']
    flat_list = list(map(helperInner, key_list))
    return tf.concat(flat_list, axis = 0)

def compile_and_save(feed_list, y_list, j):
    tsr_list = list(map(helper, feed_list))
    X = tf.ragged.stack(tsr_list).to_tensor(default_value = defaultCode)
    y = tf.ragged.stack(y_list).to_tensor(default_value = defaultCode)
    np.save(genOutPath.format(dataset, 'X_{}'.format(j)), X)
    np.save(genOutPath.format(dataset, 'y_{}'.format(j)), y)

dataset_list = {'train' : "training_data_sequenceSplit_30.tfrecord", 'val' : "validation_data_sequenceSplit_30.tfrecord", 'test' : "testing_data_sequenceSplit_30.tfrecord"}

gpus = tf.config.experimental.list_logical_devices('GPU')
#dev = '/GPU:2'
dev = gpus[1].name
tf.config.experimental.set_memory_growth(dev, True)
#gpus_str = [g.name for g in gpus]
#strategy = tf.distribute.MirroredStrategy(gpus_str[1:])

#with strategy.scope():
#with tf.device(dev):
for dataset in dataset_list.keys():

    print('\n' + dataset)
    i = 0
    j = 0

    feed_list = []
    y_list = []

    temp_data = tf.data.TFRecordDataset(os.path.join(params["tfrecords_dir"], dataset_list[dataset])).map(_parse_function)
    for data_element in temp_data:
        print('{} record {}'.format(dataset, i))

        labels = data_element[4]
        n_ligands = labels.shape[1]
        if n_ligands > 1:
            print('More than one ligand, check this out...')
            continue

        #one_hot_labels = tf.one_hot(tf.squeeze(labels) - 1, n_classes)
        y_list.append(tf.squeeze(labels))

        feed_dict = {
            'input_feat' : data_element[0],
            'rho_coords' : np.expand_dims(data_element[1], -1),
            'theta_coords' : np.expand_dims(data_element[2], -1),
            'mask' : data_element[3],
        }
        feed_list.append(feed_dict)

        i += 1

        if i % 100 == 0:
            with tf.device(dev):
                compile_and_save(feed_list, y_list, j)
                feed_list = []
                y_list = []
                j += 1


print('Finished!')
