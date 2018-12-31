import time
import os
import pandas as pd
from .utils import *

import seaborn as sns
import matplotlib.pyplot as plt

import Bullseye
from Bullseye import generate_multilogit
from Bullseye.visual import *

cwd = os.path.dirname(os.path.realpath(__file__))
result_filename = os.path.join(cwd,"data","mapfn_vs_matrix.data")

class Option:
    def __init__(self, name, phi_option, proj_option):
        self.name = name
        self.phi_option = phi_option
        self.proj_option = proj_option
        self.times = []

def mapfn_vs_matrix(recompute = False):
    if recompute:
        theta_0, x_array, y_array= generate_multilogit(d = 10, n = 10**3, k = 5)

        df = pd.DataFrame(columns=["method","times","status"])

        options = [Option("mtrx","","mapfn"),
                    Option("mapfn", "mapfn","mapfn"),
                    Option("opt. mapfn", "mapfn_opt","mapfn"),
                    Option("mtrx, proj. matrx", "",""),
                    Option("autograd mtrx", "aut_grad", "mapfn"),
                    Option("autograd mtrx, proj. matrx", "aut_grad", "")]

        n_iter = 10
        num_of_loops = 10

        for option in options:
            bull = Bullseye.Graph()
            bull.feed_with(x_array,y_array)
            bull.set_predefined_model("multilogit",
                           phi_option = option.phi_option,
                           proj_option = option.proj_option,
                           use_projections = True)
            bull.init_with(mu_0 = 0, cov_0 = 1)
            bull.build()

            for _ in range(num_of_loops):
                run_id = '{o} run {n}'.format(o = option.name, n =_)
                d = bull.run(n_iter = n_iter, run_id = run_id)
                df_ = pd.DataFrame({'method' : n_iter*[option.name],
                                    'times' : d["times"],
                                    'status': d["status"]})
                df = df.append(df_)

        with open(result_filename, "w", encoding = 'utf-8') as f:
            df.to_csv(result_filename)

    if os.path.isfile(result_filename):
        df = pd.read_csv(result_filename)
        sns.boxplot(x="method", y="times",data=df)
        plt.title('Comparison of different calculation approaches')
        handle_fig("mapfn_vs_matrix")
    else:
        raise FileNotFoundError
