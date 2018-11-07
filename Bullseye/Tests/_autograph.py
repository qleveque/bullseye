import tensorflow as tf
import numpy as np
from __future__ import print_function

from tensorflow.contrib import autograph

def square_if_positive(x):
  if x > 0:
    x = x * x
  else:
    x = 0.0
  return x

def test_np(A):
    return np.matmul(np.transpose(A),A)
    
autograph.to_code(test_np)

tf.reset_default_graph()

sqrt = autograph.to_graph(square_if_positive)
np_ = autograph.to_graph(test_np)
x = tf.get_variable("x", [], initializer = tf.initializers.constant(3), dtype = tf.float32)
vec = x = tf.get_variable("x", [3], initializer = tf.initializers.constant([2,3,1]), dtype = tf.float32)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(sqrt(x)))
    print(sess.run(np_(vec)))