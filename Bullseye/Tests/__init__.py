import os
import matplotlib
matplotlib.use('Agg')
#matplotlib.use('TkAgg')

from .utils import *

from .simple_test import simple_test
from .lm_example import lm_example
from .multilogit_example import multilogit_example

from .gradients_hessians import gradients_hessians
from .streaming_file import streaming_file
from .flatten_activations import flatten_activations
from .phi_matrix import phi_matrix
from .std_chol_svd import std_chol_svd
from .local_std_trick import local_std_trick
from .proj import proj
from .cnn import cnn
