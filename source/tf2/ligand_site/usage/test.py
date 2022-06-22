import binding
import tensorflow as tf

pdb='2VRB_AB_'
#pdb='3AYI_AB_'
#pdb='1C75_A_'

with tf.device('/CPU:0'):
	binding.predict(pdb)
