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

def multilogit_example():
    #generate data
    n=500
    k=2
    p=5*k
    theta_0, X, Y = generate_quadratic_multilogit(n=n, k=k)
    
    #create Bullseye.Graph object
    bull = Bullseye.Graph()
    bull.feed_with(X=X, Y=Y)
    bull.set_predefined_model(model = "multilogit")
    bull.set_predefined_prior(prior = "normal_iid", sigma = 10)
    bull.init_with(mu_0 = np.array([1]*p), cov_0 = 1)
    bull.build()
    
    #run graph
    d = bull.run(n_iter = 2, run_id="lm_example")
    mu = d["mu"]
    
    #interpret the result
    theta = mu_to_theta_multilogit(mu,k)
    
    df_line = draw_multilogit_separation(theta)
    df_line_0 = draw_multilogit_separation(theta_0)
    #prepare plot
    df = pd.DataFrame({'x1' : X[:,1],'x2' : X[:,2], 'y' : Y[:,0]})
    
    sns.set()
    sns.scatterplot(x = 'x1', y = 'x2', data = df,
                    hue = 'y', palette = ['r','b'],markers=['o','o'])
    sns.lineplot(x = 'x1', y = 'x2', data=df_line_0,label='θ₀')
    sns.lineplot(x = 'x1', y = 'x2', data=df_line, label = 'θ')
    handle_fig("multilogit_example")
    
    
def generate_quadratic_multilogit(n,k):
    d=5
    #theta_0 = np.random.uniform(low=-1,high=1, size = [d,k])
    theta_0 = np.asarray([[-2,20,-4,-20, 0],[0,0,0,0,0]]).T
    x_1 = np.random.uniform(size=[n])
    x_2 = np.random.uniform(size=[n])
    x_array = np.asarray([np.ones([n]),
                        x_1,
                        x_2,
                        np.square(x_1),
                        np.square(x_2)]).T
    scores = x_array@theta_0
    probs = [softmax_probabilities(score) for score in scores]
    y_array = np.asarray([np.random.multinomial(1,prob) for prob in probs])
    return theta_0, x_array, y_array.astype(np.float32)

def draw_multilogit_separation(theta):
    n = 1000
    df = pd.DataFrame(columns=['x1','x2'])
    #Xθ = Xω ⇔ X(θ-ω)=0 ⇔ Xu = 0
    u = (theta[:,1] - theta[:,0])
    for x1 in np.linspace(0,1,n):
        #u0 + u1 x1 + u3 x1^2 + u2 x2 + u4 x2^2 == 0
        #l + u2 x2 + u4 x2^2 == 0
        l = u[0] + u[1]*x1 + u[3]*(x1**2)
        x2_ = solve_2nd_order_eq(u[4],u[2],l)
        for x2 in x2_:
            if x2>0 and x2<=1:
                df_ = pd.DataFrame({'x1' : [x1], 'x2' : [x2]})
                df = df.append(df_, sort=False)
    return df

def solve_2nd_order_eq(a,b,c):
    """
    """
    if a==0:
        if b==0:
            return []
        return [-c/b]
    
    delta = b**2 - 4*a*c
    if delta < 0:
        return []
    if delta == 0:
        return [-b/(2*a)]
    else:
        ds = np.sqrt(delta)
        return [(-b+ds)/(2*a),(-b-ds)/(2*a)]
