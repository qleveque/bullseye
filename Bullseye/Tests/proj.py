import time
import os
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import Bullseye
from .utils import *

cwd = os.path.dirname(os.path.realpath(__file__))
result_filename = os.path.join(cwd,"data","local_std_trick.data")

def proj(recompute = False):
    if recompute:
        df = pd.DataFrame(columns=["method","time","status","test"])
        
        n_iter = 3
        n_loops = 2
        
        tests = ["win","not_win"]
        methods = ["proj", "without_proj"]
        
        for test in tests:
            if test=="win":
                d, n, k = (30, 500, 10)
            else:
                d, n, k = (5, 2000, 2)
                    
            theta_0, x_array, y_array = \
                    Bullseye.generate_multilogit(d = d, n = n, k = k)
            for method in methods:
                
                bull = Bullseye.Graph()
                bull.feed_with(X = x_array, Y = y_array)
                
                use_projections = (method == "proj")
                
                bull.set_predefined_model("multilogit",
                                        use_projections = use_projections)
                bull.set_predefined_prior("normal_iid", iid = True)
                bull.init_with(mu_0 = 0, cov_0 = 1)
                bull.set_options(local_std_trick = True)
                
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
        
        sns.boxplot(x="method", y="time",data=df.loc[df['test'] == 'win'])
        plt.title('Study of the interest of the local_std_trick')
        handle_fig("proj_win")
        
        sns.boxplot(x="method", y="time",data=df.loc[df['test'] != 'win'])
        plt.title('Study of the interest of the local_std_trick')
        handle_fig("proj_loses")
    else:
        raise FileNotFoundError
