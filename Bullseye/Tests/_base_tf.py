import tensorflow as tf
import numpy as np

tf.reset_default_graph()

x = tf.get_variable("x",   [2,2],   initializer = tf.initializers.constant([[1,2],[2,3]]), dtype = tf.float32)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)