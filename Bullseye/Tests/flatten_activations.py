import time
import os
import pandas as pd

import Bullseye
from Bullseye import generate_multilogit
from Bullseye.visual import *

cwd = os.path.dirname(os.path.realpath(__file__))
result_filename = os.path.join(cwd,"data","flatten_activations.data")

def flatten_activations():
    theta_0, x_array, y_array = generate_multilogit(d = 10, n = 10**4, k = 10)
    
    df = pd.DataFrame(columns=["method","times","status"])
    
    n_iter = 10
    n_loops = 10
    
    methods = ["flattening_act", "mapfn_act"]
    
    for method in methods:
        print_title(method)
        bull = Bullseye.Graph()
        bull.feed_with(X = x_array, Y = y_array)
        bull.set_model("multilogit")
        bull.init_with(mu_0 = 0, cov_0 = 1)
        
        if method == "falttening_act":
            bull.set_options(flatten_activations=True)
        else:
            bull.set_options(flatten_activations=False)
        
        bull.build()
        
        for _ in range(n_loops):
            print_subtitle('run n°{}'.format(_))
            d = bull.run(n_iter = n_iter)
            df_ = pd.DataFrame({'method' : n_iter*[method], 'times' : d["times"], 'status': d["status"]})
            df = df.append(df_)
            
    with open(result_filename, "w", encoding = 'utf-8') as f:
        df.to_csv(result_filename)