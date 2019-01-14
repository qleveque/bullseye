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

def lm_example():
    #generate data
    n=100
    theta_0, X, Y = generate_lm(d=2,n=n)
    
    #create Bullseye.Graph object
    bull = Bullseye.Graph()
    bull.feed_with(X=X, Y=Y)
    bull.set_predefined_model("LM")
    bull.set_predefined_prior("normal_iid")
    bull.init_with(mu_0 = np.array([1,1]), cov_0 = 1)
    bull.set_options(compute_hess="grad",compute_grad="act",s=500)
    bull.build()
    
    d = bull.run(n_iter = 2, run_id="lm_example")
    mu = d["mu"]
    
    #prepare plot
    df = pd.DataFrame({'x' : X[:,1], 'y' : Y})
    df_theta_0 = pd.DataFrame({'x': [0,1],
                               'y': [theta_0[0],theta_0[0]+theta_0[1]]})
    df_theta = pd.DataFrame({'x': [0,1],
                             'y': [mu[0], mu[0] + mu[1]]})
    sns.set()
    sns.scatterplot(x = 'x', y = 'y', data = df)
    sns.lineplot(x = 'x', y = 'y', data = df_theta_0, label = 'θ₀')
    sns.lineplot(x = 'x', y = 'y', data = df_theta, label = 'θ')
    
    handle_fig("lm_example")
