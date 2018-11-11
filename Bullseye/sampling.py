"""
    The ``sampling`` module
    ======================

    Contains all functions related to samplings and quadratures.
"""

import tensorflow as tf

from .utils import *

def generate_quadrature(s,k):
    """
    generate quadrature of size s for approximation of a normal law 𝒩(0,I_k)

    Keyword arguments:
        s -- sample size
        k -- dimension of the normal law
    """
    z, weights = np.polynomial.hermite.hermgauss(s)

    z_array = np.array([])
    weights_array = np.array([])

    #TODO for loop on cartesian_coords... need to find a better way
    coords = cartesian_coord(*k*[np.arange(s)])
    for coord in coords:
        new_weight = 1
        new_z = []

        for index in coord:
            new_z.append(z[index])
            new_weight*=weights[index]

        z_array.append(new_z)
        weights_array.append(new_weight)

    return z_array, weights_array

"""
def generate_sampling(s,k):
    generate sampling of size s for approximation of a normal law 𝒩(0,I_k)

    Keyword arguments:
        s -- sample size
        k -- dimension of the normal law

    z = tf.random_normal(shape=[s,k])
    z_weights = 1./s*tf.ones([s])
    return z, z_weights

"""
def generate_sampling(s,k):
    z_array = np.random.normal(size=(s,k))
    weights_array = 1./s*np.ones([s]) #TOSEE
    return z_array, weights_array

def generate_sampling_tf(s,k):
    z = tf.random_normal([s,k])
    weights = 1./s * tf.ones([s])
    return z, weights
