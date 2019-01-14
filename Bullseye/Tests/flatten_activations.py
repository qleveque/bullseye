import time
import os
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import Bullseye
from .utils import *

cwd = os.path.dirname(os.path.realpath(__file__))
result_filename = os.path.join(cwd,"data","flatten_activations.data")

def flatten_activations(recompute = False):
    ss = [50, 500]
    if recompute:
        df = pd.DataFrame(columns=["method","time","s","status"])
        
        n_iter = 5
        n_loops = 5
        
        methods = ["flatten", "tensor"]
        
        for _ in range(n_loops):
            theta_0, x_array, y_array = \
                Bullseye.generate_multilogit(d = 10, n = 1000, k = 5)
            
            for s in ss:
                for method in methods:
                    use_projections = True
                    
                    bull = Bullseye.Graph()
                    bull.feed_with(X = x_array, Y = y_array)
                    
                    bull.set_predefined_model("multilogit", use_projections = True)
                    bull.set_predefined_prior("normal_iid")
                    bull.init_with(mu_0 = 0, cov_0 = 1)
                    bull.set_options(flatten_activations =(method=="flatten_activations"),
                                    s = s)
                    
                    bull.build()
                    
                    run_id = '{method} run {_}'.format(method=method,_=_)
                    r = bull.run(n_iter = n_iter, run_id = run_id)
                    
                    df_ = pd.DataFrame({'method' : n_iter*[method], 'time' : r["times"],
                                        's': n_iter*[s], 'status' : r["status"]})
                    df = df.append(df_, sort=False)
                
        with open(result_filename, "w", encoding = 'utf-8') as f:
            df.to_csv(result_filename)
        
    if os.path.isfile(result_filename):
        df = pd.read_csv(result_filename)
        df1 = df.loc[(df['s'] == ss[0]) & (df['status']=="accepted")]
        sns.set()
        sns.boxplot(x="method", y="time",data=df1,showfliers=False)
        handle_fig("flatten_activations_0")
        
        df2 = df.loc[(df['s'] == ss[1]) & (df['status']=="accepted")]
        sns.set()
        sns.boxplot(x="method", y="time",data=df2,showfliers=False)
        handle_fig("flatten_activations_1")
    else:
        raise FileNotFoundError
