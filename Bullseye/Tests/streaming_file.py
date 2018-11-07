import time
import os
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import Bullseye
from Bullseye import generate_multilogit
from Bullseye.visual import *

cwd = os.path.dirname(os.path.realpath(__file__))
csv_filename = os.path.join(cwd,"data","streaming_file.csv")
result_filename = os.path.join(cwd,"data","streaming_file.data")

def streaming_file(recompute = False):
    if recompute:
        k=10
        theta_0, x_array, y_array =\
            generate_multilogit(d = 10, n = 10**4, k = k, file = csv_filename)
        
        df = pd.DataFrame(columns=["method","times","status"])
        
        n_iter = 10
        n_loops = 10
        
        methods = ["np_data", "streaming"]
        
        for method in methods:
            print_title(method)
            bull = Bullseye.Graph()
            
            if method=="np_data":
                bull.feed_with(X = x_array, Y = y_array)
            else:
                bull.feed_with(file=csv_filename, chunksize = 400, k = k)
                
            bull.set_model("multilogit")
            bull.init_with(mu_0 = 0, cov_0 = 1)
            bull.build()
            
            for _ in range(n_loops):
                print_subtitle('run n°{}'.format(_))
                d = bull.run(n_iter = n_iter)
                df_ = pd.DataFrame({'method' : n_iter*[method], 'times' : d["times"], 'status': d["status"]})
                df = df.append(df_)
                
        with open(result_filename, "w", encoding = 'utf-8') as f:
            df.to_csv(result_filename)
    
    if os.path.isfile(result_filename):
        df = pd.read_csv(result_filename)
        sns.boxplot(x="method", y="times",data=df)
        plt.title('Study of the efficiency of file streaming')
        plt.show()
    else:
        raise FileNotFoundError