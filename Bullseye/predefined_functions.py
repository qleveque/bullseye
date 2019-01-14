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
    return d
compute_ps["LM"] = compute_p_LM

def compute_p_CNN(d, k, conv_sizes, pools):
    tab_conv = [conv_size**2 + 1 for conv_size in conv_sizes]
    n_for_conv = sum(tab_conv)
    
    c = int(math.sqrt(d))
    flatten_size = math.floor(c/np.prod(pools))**2
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
    """
    k = Y.shape.as_list()[1]
    d = X.shape.as_list()[1]
    theta_matrix = tf.transpose(tf.reshape(theta, [k,d]))
    A = tf.matmul(X,theta_matrix)
    P=Softmax_probabilities(A)
    s = -tf.log(tf.einsum('nk,nk->n',Y,P))
    r = tf.reduce_sum(s, axis=0)
    return r

def grad_Psi_multilogit(X,Y,theta):
    k = Y.shape.as_list()[1]
    d = X.shape.as_list()[1]
    theta_matrix = tf.transpose(tf.reshape(theta, [k,d]))
    A = tf.matmul(X,theta_matrix)
    P=Softmax_probabilities(A)
    r = tf.transpose(Y-P)@X
    return -tf.reshape(r,[d*k])

def hess_Psi_multilogit(X,Y,theta):
    k = Y.shape.as_list()[1]
    d = X.shape.as_list()[1]
    theta_matrix = tf.reshape(theta, [d,k])
    A = tf.matmul(X,theta_matrix)
    P = Softmax_probabilities(A)
    P_eq = -tf.einsum('ia,ib->abi',P,tf.ones_like(P)-P)
    P_neq = tf.einsum('ia,ib->abi',P,P)
    #class -> a,b
    I = tf.eye(k)
    I_ = tf.ones([k,k])-tf.eye(k)
    big_P = tf.einsum('ab,abi->abi',I,P_eq)\
        + tf.einsum('ab,abi->abi',I_,P_neq)
    #H__ = tf.einsum('ik,abi,il->akbl',X,big_P,X)
    H_ = tf.einsum('ik,abi->akbi',X,big_P)
    H__ = tf.einsum('akbi,il->akbl',H_,X)
    H = tf.reshape(H__,[d*k,d*k])
    return -H


predefined_Psis["multilogit_without_hess"] = [Psi_multilogit, grad_Psi_multilogit,None]
predefined_Psis["multilogit_without_grad"] = [Psi_multilogit, None, hess_Psi_multilogit]
predefined_Psis["multilogit"] = [Psi_multilogit, grad_Psi_multilogit,hess_Psi_multilogit]
predefined_Psis["multilogit_simple"] = [Psi_multilogit, None,None]

def Psi_LM(X,Y,theta):
    """
    ψ(X,Y,Θ) = ψ(X,Y,β) = 0.5·n·log(2π) + 0.5 • ∑ᵢ(Yᵢ-Xᵢβ)²
    """
    #size of the sample
    n = tf.cast(tf.shape(X)[0],tf.float32)
    #e
    e = tf.square(tf.squeeze(Y,1) - tf.einsum('ij,j->i',X,theta))
    #compute the log likelihood
    log_likelihood = -0.5*n*tf.log(2*math.pi) - 0.5*tf.reduce_sum(e)
    return -log_likelihood

def grad_Psi_LM(X,Y,theta):
    """
    ∂ψ(X,Y,β)/∂βⱼ = Xⱼ•(Y-X•β)^T
    """
    #e
    e = tf.squeeze(Y,1) - tf.einsum('ij,j->i',X,theta)
    #compute the grad log likelihood
    grad_log_likelihood = tf.einsum('ij,i->j',X,e)
    return -grad_log_likelihood

def hess_Psi_LM(X,Y,theta):
    """
    ∂ψ(X,Y,β)/∂βⱼ∂βₙ = -Xⱼ•Xₙ
    """
    #compute the hess log likelihood
    hess_log_likelihood = -tf.einsum('ji,jk->ik',X,X)
    return -hess_log_likelihood

predefined_Psis["LM_simple"]=[Psi_LM, None, None]
predefined_Psis["LM_without_hess"] = [Psi_LM, grad_Psi_LM, None]
predefined_Psis["LM_without_grad"]=[Psi_LM, grad_Psi_LM, hess_Psi_LM]
predefined_Psis["LM"]=[Psi_LM, grad_Psi_LM, hess_Psi_LM]


def Psi_CNN(X,Y,theta,conv_sizes,pools):
    k = Y.shape.as_list()[1]
    P = Probabilities_CNN(X,k,theta,conv_sizes,pools)
    return -tf.reduce_sum(tf.log(tf.einsum('nk,nk->n',Y,P)),0)

predefined_Psis["CNN"] = [Psi_CNN, None, None]

#===============================================================================

predefined_Pis = {}
Pi_docstring = """
Pi functions
============

Refers to Pi_*(), grad_Pi_*() and hess_Pi_*().

Prior part of the contribution into the ψ function.

The operations used within these functions must be tensorflow
operations.

Parameters
----------
Theta : tf.tensor [n,k]
        Activation matrix.

Returns
-------
    []:
        π(A)
or
    [p]:
        ∇π(A)
or
    [p,p]:
        Hπ(A)
"""

"""
NORMAL
"""
#iid
def Pi_normal_iid(theta, mu = 0, sigma = 1):
    #ensures each variable is float
    p = tf.to_float(tf.shape(theta))
    mu = tf.to_float(mu)
    sigma = tf.to_float(sigma)
    #compute
    summands = tf.square(theta-mu*tf.ones_like(theta))
    times = -0.5/tf.square(sigma)
    const = -0.5*p* (tf.log(2*math.pi) + 2*tf.log(sigma))
    l = tf.squeeze(times*tf.reduce_sum(summands)+const)
    return -l
def grad_Pi_normal_iid(theta, mu = 0, sigma = 1):
    #ensures each variable is float
    mu = tf.to_float(mu)
    sigma = tf.to_float(sigma)
    #compute
    times = -1.0/tf.square(sigma)
    array = theta-mu*tf.ones_like(theta)
    l = times*array
    return -l
def hess_Pi_normal_iid(theta, mu = 0, sigma = 1):
    #ensures each variable is float
    sigma = tf.to_float(sigma)
    #compute
    l = - 1/tf.square(sigma)
    p = tf.shape(theta)[0]
    I = tf.eye(p)
    #return -l*tf.ones([p])
    return -l*I

predefined_Pis["normal_iid"] = [Pi_normal_iid, grad_Pi_normal_iid,
                                hess_Pi_normal_iid, True]

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
LM
"""
def Phi_LM(A,Y):
    """
    ψ(X,Y,Θ) = ψ(X,Y,β) = ∑ᵢ 0.5·(log(2π) + (Yᵢ-Xᵢβ)²)
    """
    #e
    e = tf.square(tf.squeeze(Y,1) - tf.squeeze(A,1))
    #φ(Xᵢ,β) = 0.5·(log(2π) + (Yᵢ-Xᵢβ)²)
    phi = 0.5 * (tf.log(2*math.pi) * tf.ones_like(e) + e)
    return phi

def grad_Phi_LM(A,Y):
    grad_phi = - (Y - A)
    return grad_phi

def hess_Phi_LM(A,Y):
    hess_phi = tf.expand_dims(tf.ones_like(A),2)
    return hess_phi

predefined_Phis["LM"] = [Phi_LM, grad_Phi_LM, hess_Phi_LM]
predefined_Phis["LM_without_hess"] = [Phi_LM, grad_Phi_LM, None]
predefined_Phis["LM_without_grad"] = [Phi_LM, None, hess_Phi_LM]
predefined_Phis["LM_simple"] = [Phi_LM, None, None]

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

predefined_Phis["multilogit"] = [Phi_multilogit, grad_Phi_multilogit,hess_Phi_multilogit]
predefined_Phis["multilogit_without_hess"] = [Phi_multilogit, grad_Phi_multilogit,None]
predefined_Phis["multilogit_simple"] = [Phi_multilogit, None, None]

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
    return tf.map_fn(lambda x:
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
    return tf.map_fn(lambda x:
                grad_phi_multilogit(x[0], x[1]),
                (A,Y), dtype=tf.float32)
def hess_Phi_multilogit_mapfn(A,Y):
    return tf.map_fn(lambda x:
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

def Proj_multilogit(X,k):
    #→
    d = X.shape.as_list()[1]
    X_tiled = tf.tile(X, [1,k])
    kron = np.kron(np.eye(k),np.ones((d,1)))
    KP = tf.convert_to_tensor(kron, tf.float32)
    return tf.einsum('np,pk->npk',X_tiled,KP)

predefined_Projs["multilogit"] = Proj_multilogit

def Proj_multilogit_mapfn(X, k):
    d = X.shape.as_list()[1]
    return tf.map_fn(lambda x: proj_multilogit(x,d,k), X)

predefined_Projs["multilogit_mapfn"] = Proj_multilogit_mapfn

def Proj_LM(X,k):
    return tf.expand_dims(X,2)
predefined_Projs["LM"] = Proj_LM


#===============================================================================

predefined_Predicts = {}
Predict_docstring = """
Projection functions
====================

Refers to Predict_*

The operations used within these functions must be tensorflow
operations.

Parameters
----------
X : tf.tensor [n,d]
    The design matrix.
mu : tf.tensor [p]
    The parameter matrix.

Returns
-------
tf.tensor [n]:
    Y_hat
"""

def Predict_CNN(X,theta,k,conv_sizes = None, pools = None):
    P = Probabilities_CNN(X,k,theta,conv_sizes,pools)
    return tf.math.argmax(P,axis=1)
    #return P
    
predefined_Predicts["CNN"] = Predict_CNN
