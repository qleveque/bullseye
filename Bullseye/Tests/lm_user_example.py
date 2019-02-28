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
import math


cwd = os.path.dirname(os.path.realpath(__file__))

def Psi_LM(X,Y,theta):
    n = tf.cast(tf.shape(X)[0],tf.float32)
    e = tf.square(tf.squeeze(Y,1) - tf.einsum('ij,j->i',X,theta))
    log_likelihood = -0.5*n*tf.log(2*math.pi) - 0.5*tf.reduce_sum(e)
    return -log_likelihood

def grad_Psi_LM(X,Y,theta):
    e = tf.squeeze(Y,1) - tf.einsum('ij,j->i',X,theta)
    grad_log_likelihood = tf.einsum('ij,i->j',X,e)
    return -grad_log_likelihood

def hess_Psi_LM(X,Y,theta):
    hess_log_likelihood = -tf.einsum('ji,jk->ik',X,X)
    return -hess_log_likelihood

def lm_user_example():
    n=100
    d=2
    theta_0, X, Y = generate_lm(d=d,n=n)
    
    bull = Bullseye.Graph()
    bull.feed_with(X=X, Y=Y)
    bull.set_model(Psi= Psi_LM, grad_Psi = grad_Psi_LM, hess_Psi = hess_Psi_LM, p = d)
    bull.set_predefined_prior("normal_iid")
    bull.init_with(mu_0 = 1, cov_0 = 1)
    bull.set_options(compute_hess="grad",compute_grad="act",s=500)
    bull.build()
    
    d = bull.run(n_iter = 2, run_id="lm_example")
    mu = d["mu"]
    
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
