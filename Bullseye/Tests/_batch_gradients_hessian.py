import tensorflow as tf
import numpy as np

def P_(A):
    #compute multilogit probabilities for each a of A
    sp = tf.exp(A - tf.ones(k) * tf.reduce_max(A, reduction_indices=[0])) #remove max for numerical stability + to exp
    sp = sp/tf.reduce_sum(sp, reduction_indices=[0]) #[s,n,k], where probs[s_,i] correspond to the probabilities of each a[s_,i]
    return sp

def f(x,y):
    p = P_(x)
    return -tf.log(tf.tensordot(y,p,1))
def f_grad(x,y):
    p = P_(x)
    return -y + p
def f_hess(x,y):
    p = P_(x)
    return tf.diag(p) - tf.tensordot(p,p,0)

def f_grad_(x,y):
    return tf.gradients(f(x,y),x)
def f_hess_(x,y):
    return tf.hessians(f(x,y),x)
    
tf.reset_default_graph()

x = tf.get_variable("x",   [k],   initializer = tf.initializers.constant([1,2,3,4]), dtype = tf.float32)
y =tf.get_variable("y",   [k],   initializer = tf.initializers.constant([1,0,0,0]), dtype = tf.float32)

grad = tf.gradients(f(x,y),x)
grad_ = f_grad(x,y)

hess = tf.hessians(f(x,y),x)
hess_ = f_hess(x,y)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(grad))
    print(sess.run(grad_))
    print(sess.run(hess))
    print(sess.run(hess_))