"""
→
"""

import numpy as np
import tensorflow as tf
import inspect
import math
import re

from .utils import *
from .warning_handler import *
from .predefined_functions import *

"""
CNN
"""

def apply_conv(X,W,b):
    w = int(math.sqrt(W.shape.as_list()[0]))
    #add in_channel, out_channel
    W_ = tf.expand_dims(tf.expand_dims(tf.reshape(W, [w,w]),2),2)

    X_conv = tf.nn.conv2d(
        input = X,
        filter = W_,
        strides = [1,1,1,1],
        padding = "SAME"
        )
    return X_conv

def apply_max_pool(X, pool_size):
    X_pooled = tf.layers.max_pooling2d(
        #[batch, height, width, channels]
        inputs = X,
        pool_size = pool_size,
        strides = pool_size, #→
        padding='valid'
        )
    return X_pooled

"""
MULTILOGIT
"""

def Softmax_probabilities(A):
    """
    Compute the softmax probabilities of each a in A.

    Parameters
    ----------
    A : tf.tensor [n,k]
        Score matrix.

    Returns
    -------
    tf.tensor [n,k] :
        Softmax probabilities.
    """
    #note : removing max per row (M) for numerical stability
    #compute :
    # sp_[i,j]=exp[Aᵢⱼ]
    # sp[i,j]=exp[Aᵢⱼ]/(∑ₙ exp[Aᵢₙ])

    sp_ = tf.exp(A - tf.expand_dims(tf.reduce_max(A, reduction_indices=[1]),1))
    sp = sp_/tf.expand_dims(tf.reduce_sum(sp_, reduction_indices=[1]),1)
    return sp

def softmax_probabilities(a):
    """
    Compute the softmax probabilities of a score vector a.

    Parameters
    ----------
    a : tf.tensor [k]
        Score vector.

    Returns
    -------
    tf.tensor [k]:
        Softmax probabilities.
    """
    #note : removing max for numerical stability
    #compute :
    # sp_[i]=exp[Aᵢ]
    # sp[i]=exp[Aᵢ]/(∑ₙ exp[Aₙ])
    sp_ = tf.exp(a - tf.ones_like(a) *\
                     tf.reduce_max(a, reduction_indices=[0]))
    sp = sp_/tf.reduce_sum(sp_, reduction_indices=[0]) #[n,k]
    return sp

phis = {}
phi_docstring = """
phi functions
=============

Refers to phi_*(), grad_phi_*() and hess_phi_*().

The definitions of respectively ϕᵢ, ∇ϕᵢ, Hϕᵢ when expliciting
the log posterior as ψ(θ) = ∑φᵢ(θ) = ∑ϕ(Aᵢθ).

The operations used within these functions must be tensorflow
operations.

Parameters
----------
a : tf.tensor [k]
    Activation vector.
y : tf.tensor [k]
    Response vector.

Returns
-------
    []:
        ϕ(a)
or
    [k]:
        ∇ϕ(a)
or
    [k,k]:
        Hϕ(a)
"""

"""
MULTILOGIT
"""
#no options
def phi_multilogit(a,y):
    p = softmax_probabilities(a)
    return -tf.log(tf.tensordot(y,p,1))
def grad_phi_multilogit(a,y):
    p = softmax_probabilities(a)
    return -y + p
def hess_phi_multilogit(a,y):
    p = softmax_probabilities(a)
    return tf.diag(p) - tf.tensordot(p,p,0)

#opt
def phi_multilogit_opt(a,y,p):
    return -tf.log(tf.tensordot(y,p,1))
def grad_phi_multilogit_opt(a,y,p):
    return -y + p
def hess_phi_multilogit_opt(a,y,p):
    return tf.diag(p) - tf.tensordot(p,p,0)

#aut_diff
def phi_multilogit_aut_diff(a,y):
    p = softmax_probabilities(a)
    return -tf.log(tf.tensordot(y,p,1))
def grad_phi_multilogit_aut_diff(a,y):
    return tf.gradients(phi_multilogit(a,y), a)[0]
def hess_phi_multilogit_aut_diff(a,y):
    return tf.hessians(phi_multilogit(a,y), a)[0]


projs = {}
proj_docstring = """
projection functions
====================

Refers to proj_*()

When expliciting the log posterior as ψ(θ) = ∑φᵢ(θ) = ∑ϕ(Aᵢθ),
returns Aᵢ for observation xᵢ.

The operations used within these functions must be tensorflow
operations.

Parameters
----------
x : tf.tensor [d]
    One observation of the data.

Returns
-------
tf.tensor [d, k]:
    Aᵢ corresponding to input xᵢ
"""
def proj_multilogit(x, d, k):
    x_tiled = tf.tile(x,[k])
    kp=tf.convert_to_tensor(np.kron(np.eye(k),np.ones((d,1))),
                            tf.float32)
    return tf.einsum('i,ij->ij', x_tiled, kp)
