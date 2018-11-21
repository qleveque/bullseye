"""
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

"""
FUNCTIONS TO CALL
"""
def get_predefined_psis(model, option):
    """
    When it comes to running the Bullseye algorithm directly with the log-poster
    ψ(θ), this function returns predefined ψ(θ), ∇ψ(θ) and Hψ(θ) functions.

    In this module, you will find such predefined functions in the following
    forms :
        - Psi_<model>_<option>
        - grad_Psi_<model>_<option>
        - hess_Psi_<model>_<option>

    Parameters
    ----------
    model : str
        Specifies which model should be used.
    option : str
        Specifies which option should be used.

    Returns
    -------
    Psi:
        ψ(θ)
    grad_Psi:
        ∇ψ(θ)
    hess_Psi:
        Hψ(θ)
    """
    return get_predefined("Psi",model,option)

def get_predefined_phis(model, option):
    """
    When it comes to running the Bullseye algorithm directly with the log-poster
    ψ(θ), this function returns predefined ψ(θ), ∇ψ(θ) and Hψ(θ) functions.

    In this module, you will find such predefined functions in the following
    forms :
        - Phi_<model>_<option>
        - grad_Phi_<model>_<option>
        - hess_Phi_<model>_<option>

    Parameters
    ----------
    model : str
        Specifies which model should be used.
    option : str
        Specifies which option should be used.

    Returns
    -------
    Phi:
        ϕ(θ)
    grad_Phi:
        ∇ϕ(θ)
    hess_Phi:
        Hϕ(θ)
    """
    return get_predefined("Phi",model,option)

def get_predefined_projs(model, option):
    """
    When it comes to running the Bullseye algorithm directly with the log-poster
    ψ(θ), this function returns predefined ψ(θ), ∇ψ(θ) and Hψ(θ) functions.

    In this module, you will find such predefined functions in the following
    form :
        - Proj_<model>_<option>

    Parameters
    ----------
    model : str
        Specifies which model should be used.
    option : str
        Specifies which option should be used.

    Returns
    -------
    Proj:
        Proj(xᵢ , yᵢ, θ) = Aᵢ·θ
    """
    return get_predefined("Proj",model,option)

"""
UTILS
"""
def get_predefined(part, model, option):
    """
    General function to retrieve the different functions defined in this module.
    """
    #retrieve all local functions
    m = globals().copy()
    m.update(locals())

    #retrieve the interesting suffixes wrt the given ``part`` parameter.
    suffixes = m.get(part+"_suffixes")
    #retrieve the target functions with model and option.
    f_names = [suffix+model+"_"+option for suffix in suffixes]

    #verifies that the functions are well defined
    fs = [m.get(f_name) for f_name in f_names]
    for i in range(len(fs)):
        if not fs[i]:
            err_not_implemented(f_names[i])

    #if there is only one interesting function, return it instead of the array
    if len(fs)==1:
        return fs[0]

    return fs

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

Psi_suffixes = ["Psi_","grad_Psi_","hess_Psi_"]
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

"""
MULTILOGIT
"""

def Psi_multilogit_std(X,Y,theta):
    """
    ψ(X,Y,θ) = ∑ᵢ (∑ⱼ Yⱼexp(θⱼ·xᵢ))/(∑ⱼ exp(θⱼ·xᵢ))
    We first compute Aᵢⱼ=θⱼ·xᵢ then reuse Phi_multilogit which
    computes ∑ᵢ (∑ⱼ Yⱼexp(Aᵢⱼ))/(∑ⱼ exp(Aᵢⱼ))
    """
    k = Y.shape.as_list()[1]
    d = X.shape.as_list()[1]
    theta_matrix = tf.reshape(theta, [d,k])
    A = tf.matmul(X,theta_matrix)
    return tf.reduce_sum(Phi_multilogit_std(A,Y), axis=0)

def grad_Psi_multilogit_std(X,Y,theta):
    return auto_grad_Psi(Psi_multilogit_std(X,Y,theta))

def hess_Psi_multilogit_std(X,Y,theta):
    return auto_hess_Psi(Psi_multilogit_std(X,Y,theta))

"""
CNN
"""

def Psi_CNN_std(X,Y,theta):
    #size of the sample
    n = tf.shape(X)[0]
    #image width/height
    c = int(math.sqrt(X.shape.as_list()[1]))

    #reshape X into multiple squared arrays
    X_reshaped = tf.reshape(X, [n,c,c])

    #add channel layer
    image = tf.expand_dims(X_reshaped,3)
    k = Y.shape.as_list()[1]

    #list of convolution sizes and pool sizes
    conv_sizes = [3,3]
    pools = [2,2]
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

    #for a given Score :
    #   -∑ᵢ (Score[yᵢ] - n ∑ⱼ Score[j])
    psi = - tf.reduce_sum(
                tf.einsum('ik,ik->i', Scores, Y) \
                - tf.reduce_sum(Scores, axis = 1)
                )
    return psi

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

def grad_Psi_CNN_std(X,Y,theta):
    return auto_grad_Psi(Psi_CNN_std,X,Y,theta)

def hess_Psi_CNN_std(X,Y,theta):
    return auto_hess_Psi(Psi_CNN_std,X,Y,theta)

phi_suffixes = ["phi_","grad_phi_","hess_phi_"]
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
def phi_multilogit_std(a,y):
    p = softmax_probabilities(a)
    return -tf.log(tf.tensordot(y,p,1))
def grad_phi_multilogit_std(a,y):
    p = softmax_probabilities(a)
    return -y + p
def hess_phi_multilogit_std(a,y):
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

Phi_suffixes = ["Phi_","grad_Phi_","hess_Phi_"]
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
def Phi_multilogit_std(A,Y):
    P=Softmax_probabilities(A)
    return -tf.log(tf.einsum('nk,nk->n',Y,P))
def grad_Phi_multilogit_std(A,Y):
    return -Y + Softmax_probabilities(A)
def hess_Phi_multilogit_std(A,Y):
    k = Y.get_shape().as_list()[-1]
    P=Softmax_probabilities(A)
    return tf.einsum('nk,kj->nkj', P,
                    tf.eye(k)) - tf.einsum('ni,nj->nij',P,P)

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

#mapfn
def Phi_multilogit_mapfn(A,Y):
    return tf.map_fn(lambda x:
                phi_multilogit_std(x[0], x[1]),
                (A,Y), dtype=tf.float32)
def grad_Phi_multilogit_mapfn(A,Y):
    return tf.mafn(lambda x:
                grad_phi_multilogit_std(x[0], x[1]),
                (A,Y), dtype=tf.float32)
def hess_Phi_multilogit_mapfn(A,Y):
    return tf.mafn(lambda x:
                hess_phi_multilogit_std(x[0], x[1]),
                (A,Y), dtype=tf.float32)

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

proj_suffixes = ["proj_"]
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
def proj_multilogit_std(x, d, k):
    #TODO
    x_tiled = tf.tile(x,[k])
    kp=tf.convert_to_tensor(np.kron(np.eye(k),np.ones((d,1))),
                            tf.float32)
    return tf.einsum('i,ij->ij', x_tiled, kp)

Proj_suffixes = ["Proj_"]
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

def Proj_multilogit_std(X,d,k):
    #→
    X_tiled = tf.tile(X, [1,k])
    kron = np.kron(np.eye(k),np.ones((d,1)))
    KP = tf.convert_to_tensor(kron, tf.float32)
    return tf.einsum('np,pk->npk',X_tiled,KP)

def Proj_multilogit_mapfn(X, d, k):
    return tf.map_fn(lambda x: proj_multilogit_std(x,d,k), X)

"""
HANDLE DOSCTRINGS
"""

def __set_docstrings():
    """
    Distributes properly the docstrings to the different
    functions.
    """
    m = globals().copy()
    m.update(locals())

    f_names = []
    for key, value in locals().items():
        if callable(value) and value.__module__ == __name__:
            f_names.append(key)

    to_doc = ["Psi","phi","Phi","proj","Proj"]

    for part in to_doc:
        suffixes = m.get(part + "_suffixes")
        docstring = m.get(part + "_docstring")
        suffix_regexp = "|".join([("^"+s) for s in suffixes])
        fs = [f for f in f_names if re.match(f,suffix_regexp)]
        for f in fs:
            m.get(f).__docstring__ = docstring

__set_docstrings()
