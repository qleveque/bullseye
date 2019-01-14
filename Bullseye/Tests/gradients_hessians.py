import time
import os
import pandas as pd
from .utils import *

import seaborn as sns
import matplotlib.pyplot as plt

import Bullseye
from Bullseye import generate_multilogit
from Bullseye.visual import *
from Bullseye.profilers import trace_results

cwd = os.path.dirname(os.path.realpath(__file__))
result_filename = os.path.join(cwd,"data","gradients_hessians.data")

class Option:
    def __init__(self, focus, opt):
        self.focus = focus
        self.opt = opt

def gradients_hessians(recompute = False):
    if recompute:
        
        n_iter = 10
        num_of_loops = 10
        k=3
        n=1000
        d=5
        p = d*k
        
        options = [Option("none","user"),
                   Option("grad","tf"),
                   Option("grad","act"),
                   Option("hess","tf"),
                   Option("hess","grad"),
                   Option("hess","act")]
                          
        df = pd.DataFrame(columns=["focus","opt","time","elbo"])
        
        for loop in range(num_of_loops):
            theta_0, x_array, y_array= generate_multilogit(d = d, n = n, k = k)

            for option in options:
                #psi_option
                psi_option = None if option.focus=="none" else "without_" + option.focus
                #opts
                opts = {"s" : 100}
                if option.focus == "grad":
                    opts.update({"compute_grad":option.opt})
                elif option.focus == "hess":
                    opts.update({"compute_hess":option.opt})
                
                #construction of the graph
                bull = Bullseye.Graph()
                bull.feed_with(x_array,y_array)
                bull.set_predefined_model("multilogit",
                            psi_option = psi_option)
                bull.set_predefined_prior("normal_iid")
                bull.init_with(mu_0 = 0, cov_0 = 1)
                bull.set_options(**opts)
                bull.build()
                
                run_id = 'gradients_hessians_{}_{}'.format(option.focus, option.opt)
                res = bull.run(n_iter = n_iter, run_id = run_id)
                
                times = res["times"]
                #mus,covs,elbos = trace_results(run_id)
                #thetas = [mu_to_theta_multilogit(mu,k) for mu in mus]
                #delta_thetas = [np.linalg.norm(theta_0 - theta) for theta in thetas]
                elbos = res["elbos"]
                
                df_ = pd.DataFrame({'focus' : [option.focus]*n_iter,
                                    'opt' : [option.opt]*n_iter,
                                    'time' : times,
                                    'elbo' : elbos,
                                    'iter' : list(range(n_iter))
                                    })
                
                df = df.append(df_, sort = False)

        with open(result_filename, "w", encoding = 'utf-8') as f:
            df.to_csv(result_filename)

    if os.path.isfile(result_filename):
        df = pd.read_csv(result_filename)
        df_ = df.loc[df['iter']!=0]
        
        #gradients
        df_grad = df_.loc[(df['focus']=="grad") | (df['focus']=="none")]

        sns.set()
        sns.boxplot(x="opt", y="time",data=df_grad)
        handle_fig("gradients_time")
        
        sns.set()
        sns.lineplot(x="iter", y="elbo",hue="opt",data=df_grad)
        handle_fig("gradients_prec")
        
        #hessians
        df_hess = df_.loc[(df['focus']=="hess") | (df['focus']=="none")]
        
        sns.set()
        sns.boxplot(x="opt", y="time",data=df_hess)
        handle_fig("hessians_time")
        
        sns.set()
        sns.lineplot(x="iter", y="elbo",hue="opt",data=df_hess)
        handle_fig("hessians_prec")
    else:
        raise FileNotFoundError
