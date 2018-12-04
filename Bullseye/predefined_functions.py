"""
→→→
The ``predefined_functions`` module
===================================

Contains all functions related to the computation of the
log-posterior ψ(θ) = ∑φᵢ(θ) = ∑ϕᵢ(Aᵢθ).
In particular, contains the definition of predefined ψ's,
ϕᵢ's, their gradients, hessians, and optionally their
projection matrices Aᵢ for different models.

The public function that may be used are
``get_predefined_phis``, ``get_predefined_projs`` and
``get_predefined_psis``.

:Example:

>>> phi, grad_phi, hess_phi = \
>>>      get_predefined_functions("multilogit")
>>> A = get_predefined_projs("multilogit")
"""

import numpy as np
import tensorflow as tf
import inspect
import math
import re

from .utils import *
from .warning_handler import *
from .predefined_functions_aux import *

#===============================================================================
compute_ps = {}
compute_p_docstring = """
compute_p functions
===================
"""

def compute_p_multilogit(d,k):
    return d*k
compute_ps["multilogit"] = compute_p_multilogit

def compute_p_LM(d,k):
    return d+1
compute_ps["LM"] = compute_p_LM

def compute_p_CNN(d, k, conv_sizes, pools):
    n_for_conv = sum([conv_size**2 + 1 for conv_size in conv_sizes])
    c = int(math.sqrt(d))
    flatten_size = int(c/np.prod(pools))**2
    n_for_multilogit = flatten_size * k + k
    return n_for_conv + n_for_multilogit
compute_ps["CNN"] = compute_p_CNN

#===============================================================================
predefined_Psis = {}
Psi_docstring = """
Psi functions
=============

Refers to Psi_*(), grad_Psi_*() and hess_Psi_*().
The definitions of ψ, ∇ψ, Hψ when considering the log posterior as ψ(θ)
The operations used within these functions must be tensorflow operations.

Parameters
----------
X : tf.tensor [n,d]
    Design matrix.
Y : tf.tensor [n,k]
    Response matrix.
theta : tf.tensor [p]
    θ

Returns
-------
    []:
        ψ(θ)
or
    [p]:
        ∇ψ(θ)
or
    [p,p]:
        Hψ(θ)
"""

def Psi_multilogit(X,Y,theta):
    """
    ψ(X,Y,θ) = - log[ ∑ᵢ (∑ⱼ Yⱼexp(θⱼ·xᵢ))/(∑ⱼ exp(θⱼ·xᵢ)) ]
    We first compute Aᵢⱼ=θⱼ·xᵢ then reuse Phi_multilogit which
    computes ∑ᵢ (∑ⱼ Yⱼexp(Aᵢⱼ))/(∑ⱼ exp(Aᵢⱼ))
    """
    k = Y.shape.as_list()[1]
    d = X.shape.as_list()[1]
    theta_matrix = tf.reshape(theta, [d,k])
    A = tf.matmul(X,theta_matrix)
    return tf.reduce_sum(Phi_multilogit(A,Y), axis=0)
predefined_Psis["multilogit"] = [Psi_multilogit, None, None]

def Psi_LM(X,Y,theta):
    """
    ψ(X,Y,Θ) = ψ(X,Y,β,σ²) = -0.5·n·[log(2π) + σ²] - 0.5 * 1/σ² ∑ᵢ(Yᵢ-Xᵢβ)²
    """
    #size of the sample
    #→→→ log sigma
    n = tf.cast(tf.shape(X)[0],tf.float32)
    d = X.shape.as_list()[1]
    #split θ into β and σ
    beta, sigma_squared = tf.split(theta, [d,1])
    #e
    e = tf.square(tf.squeeze(Y,1) - tf.einsum('ij,j->i',X,beta))

    #compute the log likelihood
    log_likelihood = -0.5*n*(tf.log(2*math.pi)+tf.log(sigma_squared)) \
                - 0.5 * 1/sigma_squared * tf.reduce_sum(e)
    #print(-tf.squeeze(log_likelihood))
    return -tf.squeeze(log_likelihood)
predefined_Psis["LM"]=[Psi_LM, None, None]

def Psi_CNN(X,Y,theta,conv_sizes,pools):
    #size of the sample
    n = tf.shape(X)[0]
    #image width/height
    c = int(math.sqrt(X.shape.as_list()[1]))

    #reshape X into multiple squared arrays
    X_reshaped = tf.reshape(X, [n,c,c])

    #add channel layer
    image = tf.expand_dims(X_reshaped,3)
    k = Y.shape.as_list()[1]

    #size of the final flatten list
    flatten_size = int(c/np.prod(pools))**2

    #split theta accordingly to the convolution filters
    how_to_split_theta = []
    for i in range(len(conv_sizes)):
        how_to_split_theta += [conv_sizes[i]**2, 1]

    #for final logistic regression
    how_to_split_theta += [flatten_size*k, k]

    #finally split theta
    theta_splits = tf.split(theta, how_to_split_theta)

    for i in range(len(conv_sizes)):
        #retrieve current W and b
        W = theta_splits[2*i]
        b = theta_splits[2*i + 1]

        #apply the convolutions and the max pools
        image = apply_conv(image,W,b)
        image = apply_max_pool(image,pools[i])

    #flatten
    flat = tf.layers.Flatten()(image) # of size [n, flatten_size]

    #log multilogit on what remains and as we compute the log,
    # we don't use exp
    W = tf.reshape(theta_splits[-2],[flatten_size,k])
    b = theta_splits[-1]
    #compute scores
    Scores = tf.expand_dims(b,0) + flat@W

    #compute probabilities from Scores
    P=Softmax_probabilities(Scores)
    return -tf.reduce_sum(tf.log(tf.einsum('nk,nk->n',Y,P)),0)
predefined_Psis["CNN"] = [Psi_CNN, None, None]

#===============================================================================

predefined_Phis = {}
Phi_docstring = """
Phi functions
=============

Refers to Phi_*(), grad_Phi_*() and hess_Phi_*().

Matrix form of the functions phi_*(), grad_phi_*() ans
hess_phi_*(). In particular :
    Phi_*(A) = {phi_*(aᵢ) for aᵢ ∈ A}.

The operations used within these functions must be tensorflow
operations.

Parameters
----------
A : tf.tensor [n,k]
    Activation matrix.
Y : tf.tensor [n,k]
    Response matrix.

Returns
-------
    [n]:
        ϕ(A)
or
    [n,k]:
        ∇ϕ(A)
or
    [n,k,k]:
        Hϕ(A)
"""

"""
MULTILOGIT
"""
#no options
def Phi_multilogit(A,Y):
    P=Softmax_probabilities(A)
    return -tf.log(tf.einsum('nk,nk->n',Y,P))
def grad_Phi_multilogit(A,Y):
    return -Y + Softmax_probabilities(A)
def hess_Phi_multilogit(A,Y):
    k = Y.get_shape().as_list()[-1]
    P=Softmax_probabilities(A)
    return tf.einsum('nk,kj->nkj', P,
                    tf.eye(k)) - tf.einsum('ni,nj->nij',P,P)

predefined_Phis["multilogit"] = [Phi_multilogit,
                    grad_Phi_multilogit,
                    hess_Phi_multilogit]

#aut_grad
def Phi_multilogit_aut_grad(A,Y):
    return -tf.log(tf.einsum('nk,nk->n', Y,
                            Softmax_probabilities(A)))
def grad_Phi_multilogit_aut_grad(A,Y):
    return tf.gradients(Phi_multilogit(A,Y),A)[0]
def hess_Phi_multilogit_aut_grad(A,Y):
    k = Y.get_shape().as_list()[-1]
    P=Softmax_probabilities(A)
    return tf.einsum('nk,kj->nkj',P,
                    tf.eye(k)) - tf.einsum('ni,nj->nij',P,P)

predefined_Phis["multilogit_aut_grad"] = [Phi_multilogit_aut_grad,
                    grad_Phi_multilogit_aut_grad,
                    hess_Phi_multilogit_aut_grad]

#mapfn_opt
def Phi_multilogit_mapfn_opt(A,Y):
    P=Softmax_probabilities(A)
    return tf.map_fn(lambda x:
                    phi_multilogit_opt(x[0], x[1], x[2]),
                    (A,Y,P), dtype=tf.float32)
def grad_Phi_multilogit_mapfn_opt(A,Y):
    P=Softmax_probabilities(A)
    return tf.map_fn(lambda x:
                    grad_phi_multilogit_opt(x[0], x[1], x[2]),
                    (A,Y,P), dtype=tf.float32)
def hess_Phi_multilogit_mapfn_opt(A,Y):
    P=Softmax_probabilities(A)
    return tf.mafn(lambda x:
                hess_phi_multilogit_opt(x[0], x[1], x[2]),
                (A,Y,P), dtype=tf.float32)

predefined_Phis["multilogit_mapfn_opt"] = [Phi_multilogit_mapfn_opt,
                    grad_Phi_multilogit_mapfn_opt,
                    hess_Phi_multilogit_mapfn_opt]

#mapfn
def Phi_multilogit_mapfn(A,Y):
    return tf.map_fn(lambda x:
                phi_multilogit(x[0], x[1]),
                (A,Y), dtype=tf.float32)
def grad_Phi_multilogit_mapfn(A,Y):
    return tf.mafn(lambda x:
                grad_phi_multilogit(x[0], x[1]),
                (A,Y), dtype=tf.float32)
def hess_Phi_multilogit_mapfn(A,Y):
    return tf.mafn(lambda x:
                hess_phi_multilogit(x[0], x[1]),
                (A,Y), dtype=tf.float32)

predefined_Phis["multilogit_mapfn"] = [Phi_multilogit_mapfn,
                    grad_Phi_multilogit_mapfn,
                    hess_Phi_multilogit_mapfn]

#mapfn_aut_diff
def Phi_multilogit_mapfn_aut_diff(A,Y):
    return tf.map_fn(lambda x:
                phi_multilogit_aut_diff(x[0], x[1]),
                (A,Y), dtype=tf.float32)
def grad_Phi_multilogit_mapfn_aut_diff(A,Y):
    return tf.map_fn(lambda x:
                grad_phi_multilogit_aut_diff(x[0], x[1]),
                (A,Y), dtype=tf.float32)
def hess_Phi_multilogit_mapfn_aut_diff(A,Y):
    return tf.map_fn(lambda x:
                hess_phi_multilogit_aut_diff(x[0], x[1]),
                (A,Y), dtype=tf.float32)

predefined_Phis["multilogit_mapfn_aut_diff"] = [Phi_multilogit_mapfn_aut_diff,
                    grad_Phi_multilogit_mapfn_aut_diff,
                    hess_Phi_multilogit_mapfn_aut_diff]

#===============================================================================

predefined_Projs = {}
Proj_docstring = """
Projection functions
====================

Refers to Proj_*

The matrix form of the projections proj_*()
Corresponds to (proj_*(xᵢ) : i∈〚1,n〛).

The operations used within these functions must be tensorflow
operations.

Parameters
----------
X : tf.tensor [n,d]
    The design matrix.

Returns
-------
tf.tensor [n, d, k]:
    {Aᵢ : i∈〚1,n〛}
"""

def Proj_multilogit(X,d,k):
    #→
    X_tiled = tf.tile(X, [1,k])
    kron = np.kron(np.eye(k),np.ones((d,1)))
    KP = tf.convert_to_tensor(kron, tf.float32)
    return tf.einsum('np,pk->npk',X_tiled,KP)

predefined_Projs["multilogit"] = Proj_multilogit

def Proj_multilogit_mapfn(X, d, k):
    return tf.map_fn(lambda x: proj_multilogit(x,d,k), X)

predefined_Projs["multilogit_mapfn"] = Proj_multilogit_mapfn
