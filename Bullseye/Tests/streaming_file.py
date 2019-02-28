import time
import os
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import Bullseye
from .utils import *

cwd = os.path.dirname(os.path.realpath(__file__))
csv_filename = os.path.join(cwd,"data","streaming_file.csv")
result_filename = os.path.join(cwd,"data","streaming_file.data")

def streaming_file(recompute = True):
    ms = [5000,1000]
    if recompute:
        k=5
        d=10
        n=50000
        theta_0, x_array, y_array =\
            Bullseye.generate_multilogit(d = d, n = n, k = k, file = csv_filename)
        
        df = pd.DataFrame(columns=["method","time","status"])
        
        n_iter = 3
        n_loops = 3
        methods = ["pandas", "tf"]
        
        for m in ms:
            for method in methods:
                bull = Bullseye.Graph()
                bull.feed_with(file=csv_filename, k = k, m=m)
                bull.set_predefined_model("multilogit")
                bull.set_predefined_prior("normal_iid")
                bull.init_with(mu_0 = 0, cov_0 = 1)
                bull.set_options(tf_dataset=(method=="tf"),to_one_hot=True)
                bull.build()
                
                for _ in range(n_loops):
                    run_id = '{method} run {n}'.format(n=_, method=method)
                    d = bull.run(n_iter = n_iter, run_id = run_id)
                    df_ = pd.DataFrame({'method' : n_iter*[method],
                                        'time' : d["times"],
                                        'status': d["status"],
                                        'm' : n_iter*[m]})
                    df = df.append(df_, sort = False)
                
        with open(result_filename, "w", encoding = 'utf-8') as f:
            df.to_csv(result_filename)
    
    if os.path.isfile(result_filename):
        df = pd.read_csv(result_filename)
        sns.set()
        sns.boxplot(x="method", y="time",data=df.loc[df["m"]==ms[0]])
        handle_fig("streaming_file_0")
        sns.set()
        sns.boxplot(x="method", y="time",data=df.loc[df["m"]==ms[1]])
        handle_fig("streaming_file_1")
    else:
        raise FileNotFoundError
