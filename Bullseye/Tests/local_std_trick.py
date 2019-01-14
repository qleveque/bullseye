import time
import os
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import Bullseye
from .utils import *

cwd = os.path.dirname(os.path.realpath(__file__))
result_filename = os.path.join(cwd,"data","local_std_trick.data")

def local_std_trick(recompute = False):
    if recompute:
        df = pd.DataFrame(columns=["method","time","status","test"])
        
        n_iter = 3
        n_loops = 2
        
        tests = ["win","not_win"]
        methods = ["local_std_trick", "without_lst"]
        
        for test in tests:
            if test=="win":
                d, n, k = (30, 1000, 10)
            else:
                d, n, k = (5, 200000, 2)
                    
            theta_0, x_array, y_array = \
                    Bullseye.generate_multilogit(d = d, n = n, k = k)
            for method in methods:
                
                bull = Bullseye.Graph()
                bull.feed_with(X = x_array, Y = y_array)
                bull.set_predefined_model("multilogit",
                                        use_projections = True)
                bull.set_predefined_prior("normal_iid")
                bull.init_with(mu_0 = 0, cov_0 = 1)
                
                local_std_trick = (method=="local_std_trick")

                bull.set_options(prior_iid = True, local_std_trick = local_std_trick)
                
                bull.build()
                
                for _ in range(n_loops):
                    run_id = '{method} run {n}'.format(method=method,n=_)
                    d = bull.run(n_iter = n_iter, run_id = run_id)
                    df_ = pd.DataFrame({'method' : n_iter*[method], 'time' : d["times"],
                                        'status': d["status"], 'test' : n_iter*[test]})
                    df = df.append(df_, sort=False)
                
        with open(result_filename, "w", encoding = 'utf-8') as f:
            df.to_csv(result_filename)
        
    if os.path.isfile(result_filename):
        df = pd.read_csv(result_filename)
        
        sns.set()
        sns.boxplot(x="method", y="time",data=df.loc[df["test"] == "win"])
        handle_fig("local_std_trick_win")
        
        sns.set()
        sns.boxplot(x="method", y="time",data=df.loc[df["test"] == "not_win"])
        handle_fig("local_std_trick_loses")
    else:
        raise FileNotFoundError
