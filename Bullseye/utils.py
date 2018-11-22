"""
    The ``utils`` module
    ======================

    Contains various functions useful for the Bullseye algorithm.
"""

import numpy as np
import struct
import sys
import tensorflow as tf
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

def partition_list(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]

def softmax_probabilities(z):
    """
    return the softmax probabilities of z

    Keyword arguments:
        z -- np.array() or list() containing the different initial scores
    """
    #for numerical stability
    z = z - np.max(z)
    #unnormalized probabilities
    P = np.exp(z)
    return P/np.sum(P)

def generate_multilogit(d,n,k, file = None):
    """
    generate softmax data

    Keyword arguments:
        d -- dimension of θ for one class
        n -- number of observations
        k -- number of class for softmax
    """

    p=d*k

    #the θᵢ's related to class k will be in i_[k]
    i_ = list(partition_list(range(p), d))

    #create θ₀ matrix (with k class in d dimensions)
    theta_0 = np.float32( 0.3 * np.random.randn(k,d) / (k*d)**0.5 )
    #flatten θ₀
    theta_0 = np.ndarray.flatten(theta_0)

    #random design matrix
    x_array = np.random.randn(n,d)
    #intercepts
    x_array[:,0] = 1
    #compute scores
    scores = np.transpose([x_array@theta_0[i] for i in i_]) #TODO

    #compute probs
    probs = [softmax_probabilities(score) for score in scores]
    #generate labels
    y_array = np.asarray([np.random.multinomial(1,prob) for prob in probs])

    if file is not None:
        y_flat = np.expand_dims(from_one_hot(y_array), 1)
        to_write = np.hstack((y_flat,x_array))
        np.savetxt(file, to_write, delimiter=",", fmt='% 1.3f')

    return theta_0, x_array.astype(np.float32), y_array.astype(np.float32)

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

def to_one_hot(y):
    y = list(map(int,y))
    n_values = np.max(y) + 1
    return np.eye(n_values)[y]

def from_one_hot(y):
    return np.asarray([np.where(r==1)[0][0] for r in y])

def sublist(a, b):
    return set(a) <= set(b)

"""
def decode_csv(line):
   parsed_line = tf.decode_csv(line, record_defaults)
   label =  parsed_line[-1]
   # label is the last element of the list
   del parsed_line[-1]
   # delete the last element from the list
   del parsed_line[0]
   # even delete the first element bcz it is assumed NOT to be a feature
   features = tf.stack(parsed_line)
   # Stack features so that you can later vectorize forward prop., etc.
   #label = tf.stack(label)
   #NOT needed. Only if more than 1 column makes the label...
   batch_to_return = features, label
   return batch_to_return
"""

def matrix_sqrt(A):
    s, u, v = tf.linalg.svd(A)
    s_sqrt = tf.linalg.diag(tf.sqrt(s))
    r = tf.matmul(u, tf.matmul(s_sqrt, v, adjoint_b=True))
    return r

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
