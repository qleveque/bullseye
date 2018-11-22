import os
import matplotlib
matplotlib.use('Agg')
#matplotlib.use('TkAgg')

from .utils import *

from .simple_test import simple_test

from .mapfn_vs_matrix import mapfn_vs_matrix
from .streaming_file import streaming_file
from .flatten_activations import flatten_activations
