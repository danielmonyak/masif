import tensorflow as tf

# Edited by Daniel Monyak
# Changed tf.FixedLenFeature to tf.io.FixedLenFeature
# Changed tf.VarLenFeature to tf.io.VarLenFeature
# Changed tf.parse_single_example to tf.io.parse_single_example
# Changed tf.sparse_tensor_to_dense to tf.compat.v1.sparse_tensor_to_dense

def _parse_function(example_proto):
    keys_to_features = {
        "input_feat_shape": tf.io.FixedLenFeature([3], dtype=tf.int64),
        "input_feat": tf.io.VarLenFeature(dtype=tf.float32),
        "rho_wrt_center_shape": tf.io.FixedLenFeature([2], dtype=tf.int64),
        "rho_wrt_center": tf.io.VarLenFeature(dtype=tf.float32),
        "theta_wrt_center_shape": tf.io.FixedLenFeature([2], dtype=tf.int64),
        "theta_wrt_center": tf.io.VarLenFeature(dtype=tf.float32),
        "mask_shape": tf.io.FixedLenFeature([3], dtype=tf.int64),
        "mask": tf.io.VarLenFeature(dtype=tf.float32),
        "pdb": tf.io.FixedLenFeature([], dtype=tf.string),
        "pocket_labels_shape": tf.io.FixedLenFeature([2], dtype=tf.int64),
        "pocket_labels": tf.io.VarLenFeature(dtype=tf.int64),
    }
    parsed_features = tf.io.parse_single_example(example_proto, keys_to_features)
    input_feat = tf.compat.v1.sparse_tensor_to_dense(parsed_features["input_feat"])
    input_feat = tf.reshape(
        input_feat, tf.cast(parsed_features["input_feat_shape"], tf.int32)
    )
    rho_wrt_center = tf.compat.v1.sparse_tensor_to_dense(parsed_features["rho_wrt_center"])
    rho_wrt_center = tf.reshape(
        rho_wrt_center, tf.cast(parsed_features["rho_wrt_center_shape"], tf.int32)
    )
    theta_wrt_center = tf.compat.v1.sparse_tensor_to_dense(parsed_features["theta_wrt_center"])
    theta_wrt_center = tf.reshape(
        theta_wrt_center, tf.cast(parsed_features["theta_wrt_center_shape"], tf.int32)
    )
    mask = tf.compat.v1.sparse_tensor_to_dense(parsed_features["mask"])
    mask = tf.reshape(mask, tf.cast(parsed_features["mask_shape"], tf.int32))
    labels = tf.compat.v1.sparse_tensor_to_dense(parsed_features["pocket_labels"])
    labels = tf.reshape(
        labels, tf.cast(parsed_features["pocket_labels_shape"], tf.int32)
    )
    return (
        input_feat,
        rho_wrt_center,
        theta_wrt_center,
        mask,
        labels,
        parsed_features["pdb"],
    )
