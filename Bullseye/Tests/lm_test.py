import Bullseye
import tensorflow as tf
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import time
import pandas as pd
from tensorflow.initializers import constant as tic
from .utils import *
from Bullseye.visual import *
from Bullseye.profilers import trace_results

cwd = os.path.dirname(os.path.realpath(__file__))
data_filename = os.path.join(cwd,"data","ws_const.csv")

def lm_test():
    bull = Bullseye.Graph()

    data = np.genfromtxt(data_filename, delimiter=',')
    X = data[:,1:]
    Y = data[:,0]
    bull.feed_with(X = X, Y = Y)

    bull.set_predefined_model(model = "LM",
                            use_projections = False)
    bull.set_predefined_prior(prior = "normal_iid", sigma = 5.)

    bull.init_with(mu_0 = np.array([0.5,0.5]), cov_0 = 1)

    options = {
        "speed" : 1,
        "keep_track" : True,
        "tf_dataset" : True,
        "s" : 100,
        "comp_opt" : "cholesky",
        "backtracking_degree" : 1,
        "compute_gamma": False
    }

    bull.set_options(**options)

    bull.build()

    d = bull.run(n_iter = 3, run_id="lm", 
                debug_array=["new_cov","new_cov_sqrt","new_mu","computed_e",
                            "computed_e_prior", "new_logdet", "computed_beta", "computed_rho"])

    mus, covs, elbos = trace_results("lm")

    a = 1.20
    b = 2.60
    mu_from = [a, b]
    
    colors = np.random.rand(100)
    plt.scatter(X[:,1], Y, c=colors, alpha=0.5)
    
    for idx,mu in enumerate(mus):
        mu_to = [mu[0] + a*mu[1], mu[0] + b*mu[1]]
        alpha = 1-float(idx+1)/(len(mus)+1)
        plt.plot(mu_from, mu_to, 'k-', lw=2,alpha=alpha)
    
    plt.title('Linear model test')
    handle_fig("lm")

    print(list(d))
