"""
    The ``utils`` module
    ======================

    Contains various functions useful for the Bullseye algorithm.
"""

import numpy as np
import struct
import sys
import tensorflow as tf
import pandas as pd

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

def partition_list(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]

"""
MULTILOGIT
"""
    
def mu_to_theta_multilogit(mu,k):
    d = int(mu.shape[0]/k)
    theta_hat = np.ndarray.reshape(mu, (k,d)).T
    #theta_hat = np.ndarray.reshape(mu,(d,k))
    return theta_hat
    
def test_multilogit(theta,X):
    S = X@theta
    P = [softmax_probabilities(s) for s in S]
    R = [np.argmax(p) for p in P]
    return R
    
def evaluate_multilogit_results(theta_0, mu):
    d, k = theta_0.shape
    theta_hat = mu_to_theta_multilogit(mu,k)
    par = {'axis': 1, 'keepdims' : True}
    theta_hat_r = theta_hat - np.max(theta_hat, **par) \
            + np.max(theta_0, **par)
    e = theta_0 - theta_hat_r
    print("theta_0")
    print(theta_0)
    print("theta_hat")
    print(theta_hat_r)
    print("error")
    print(e)
    print("norm of the error")
    print(np.linalg.norm(e))
    
    
def generate_multilogit(d,n,k, file = None):
    """
    →
    """
    p=d*k
    #create θ₀ matrix (with k class in d dimensions)
    theta_0 = np.random.randn(d,k)/(k*d)**0.5
    #random design matrix
    x_array = np.random.randn(n,d)
    #intercepts
    x_array[:,0] = 1.
    #compute scores
    scores = x_array@theta_0
    #compute probs
    probs = [softmax_probabilities(score) for score in scores]
    #generate labels
    y_array = np.asarray([np.random.multinomial(1,prob) for prob in probs])

    if file is not None:
        y_flat = np.expand_dims(from_one_hot(y_array), 1)
        to_write = np.hstack((y_flat,x_array))
        np.savetxt(file, to_write, delimiter=",", fmt='% 1.3f')

    return theta_0, x_array, y_array.astype(np.float32)

def softmax_probabilities(z):
    #for numerical stability
    z = z - np.max(z)
    #unnormalized probabilities
    P = np.exp(z)
    return P/np.sum(P)

"""
LINEAR MODELS
"""

def generate_lm(d,n):
    theta_0 = np.random.uniform(low=1.0, high=2.0,size = [d])
    
    #random design matrix
    x_array = np.random.uniform(size = [n,d])
    #intercepts
    x_array[:,0] = 1.
    #compute scores
    scores = x_array@theta_0
    #normal error
    e = np.random.normal(scale = 0.2,size = [n])
    #y_array
    y_array = x_array@theta_0 + e
    
    return theta_0, x_array, y_array

"""
CNN
"""

def cartesian_coord(*arrays):
    """
    source : https://stackoverflow.com/questions/1208118/using-numpy-to-build-a\
    n-array-of-all-combinations-of-two-arrays
    """
    grid = np.meshgrid(*arrays)
    coord_list = [entry.ravel() for entry in grid]
    points = np.vstack(coord_list).T
    return points

def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

def to_one_hot(y,k):
    y = list(map(int,y))
    return np.eye(k)[y]

def from_one_hot(y):
    return np.asarray([np.where(r==1)[0][0] for r in y])

def sublist(a, b):
    return set(a) <= set(b)

def matrix_sqrt(A):
    #s, u, v = tf.linalg.svd(A)
    #s_sqrt = tf.linalg.diag(tf.sqrt(s))
    #r = tf.matmul(u, tf.matmul(s_sqrt, v, adjoint_b=True))
    r = tf.transpose(tf.linalg.cholesky(A))
    return r

"""
Color handling
"""
if sys.platform not in ['win32', 'Windows']:
    class bcolors:
        HEADER = '\033[95m'
        OKBLUE = '\033[94m'
        OKGREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'
else:
    class bcolors:
        HEADER = ''
        OKBLUE = ''
        OKGREEN = ''
        WARNING = ''
        FAIL = ''
        ENDC = ''
        BOLD = ''
        UNDERLINE = ''

class NullIO(StringIO):
    def write(self, txt):
       pass
