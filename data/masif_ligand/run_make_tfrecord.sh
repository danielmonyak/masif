./make_tfrecord.slurm > make_tf.out 2>make_tf.err &
disown -h $!
echo $! > make_tfrecord_pid.txt
