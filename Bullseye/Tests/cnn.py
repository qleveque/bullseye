import time
import os
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow.initializers import constant as tic
from Bullseye.predefined_functions_aux import Probabilities_CNN

import Bullseye
from .utils import *
import pandas as pd

cwd = os.path.dirname(os.path.realpath(__file__))
result_filename = os.path.join(cwd,"data","cnn.data")
data_file = "data/norm_mnist.csv"
k = 10
d = 28
conv_sizes = [3,3,3]
pools = [2,2,2]

def cnn(recompute = False):
    if recompute:
        bull = Bullseye.Graph()
        
        bull.feed_with(file = data_file, m=1000,k=k,to_one_hot=True)
        bull.set_predefined_model(model = "CNN",
                                conv_sizes = conv_sizes,
                                pools = pools)
        bull.set_predefined_prior("normal_iid", mu = 0, sigma = 1)

        bull.init_with(mu_0 = 0.3, cov_0 = 0.001)

        options = {
            "chunk_as_sum" : True,
            "speed" : 0.1,
            "compute_hess" : "act",
            "compute_grad" : "act",
            "diag_cov" : True,
            "s":3,
            "brutal_iteration":True,
            "keep_track": True
        }

        bull.set_options(**options)
        bull.build()
        res = bull.run(n_iter = 10, run_id="cnn")
        
    if os.path.isfile(result_filename):
        mu, cov, elbo = Bullseye.read_results("cnn")
        data = np.asarray(pd.read_csv(data_file, sep=',', nrows = 3000))
        
        x_array = data[:,1:]
        y_array = to_one_hot(data[:,0],k)
        
        bull = Bullseye.Graph()
        T = bull.predict(x_array,mu,k, model="CNN",
                         conv_sizes = conv_sizes,
                         pools = pools)
        
        print(T)
        print(np.max(T))
        
    else:
        raise FileNotFoundError
