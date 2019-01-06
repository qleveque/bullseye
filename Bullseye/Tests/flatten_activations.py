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
    if recompute:
        theta_0, x_array, y_array = \
                Bullseye.generate_multilogit(d = 4, n = 1000, k = 2)
        
        df = pd.DataFrame(columns=["method","times","status"])
        
        n_iter = 10
        n_loops = 1
        
        #methods = ["flattening_act", "mapfn_act"]
        methods = ["proj"]
        
        for method in methods:
            use_projections = False
            #option = "without_hess"
            #option= "spe"
            option = None
            #option = "simple"
            
            options = {"s" : 100, "speed" : 1,
                       "flatten_activations" : False,
                       "compute_hess" : "tf",
                       "compute_grad": "tf",
                       "comp_opt" : "cholesky",
                       "local_std_trick": True}
            
            bull = Bullseye.Graph()
            bull.feed_with(X = x_array, Y = y_array)
            
            bull.set_predefined_model("multilogit", phi_option = option, psi_option = option,
                                     use_projections = use_projections)
            bull.set_predefined_prior("normal_iid")
            bull.init_with(mu_0 = 0, cov_0 = 1)
            
            #debug_array=["computed_e", "computed_e_prior", "new_logdet", "computed_beta", "computed_rho"]
            debug_array = []
            
            bull.set_options(**options)
            
            bull.build()
            
            for _ in range(n_loops):
                run_id = '{method} run {n}'.format(method=method,n=_)
                d = bull.run(n_iter = n_iter, run_id = run_id, debug_array = debug_array)
                Bullseye.evaluate_multilogit_results(theta_0, d["mu"])
                
                df_ = pd.DataFrame({'method' : n_iter*[method], 'times' : d["times"], 'status': d["status"]})
                df = df.append(df_, sort=False)
                
        with open(result_filename, "w", encoding = 'utf-8') as f:
            df.to_csv(result_filename)
        
    if os.path.isfile(result_filename):
        df = pd.read_csv(result_filename)
        sns.boxplot(x="method", y="times",data=df.loc[df['status'] == 'accepted'])
        plt.title('Study of the interest of flattening activations')
        handle_fig("flatten_activations")
    else:
        raise FileNotFoundError
