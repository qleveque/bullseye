import time
import os
import pandas as pd
from .utils import *

import seaborn as sns
import matplotlib.pyplot as plt

import Bullseye
from Bullseye import generate_multilogit
from Bullseye.visual import *

from Bullseye.predefined_functions import predefined_Phis

cwd = os.path.dirname(os.path.realpath(__file__))
result_filename = os.path.join(cwd,"data","phi_matrix.data")

def phi_matrix(recompute = True):
    df = pd.DataFrame(columns=["option","time","n"])
    if recompute:
        #initialize variables
        k = 5
        ns = [10**i for i in [3,4]]
        model = "multilogit"
        options = ["matrix","mapfn","mapfn_opt"]
        nb_iter = 10
        
        for n in ns:
            print_title("test for n = {}".format(n))
            
            #create tensorflow graph
            tf.reset_default_graph()
            A = tf.get_variable("A",[n,k],
                                initializer = tf.zeros_initializer,
                                dtype = tf.float32)
            Y = tf.get_variable("Y",[n,k],
                                initializer = tf.zeros_initializer,
                                dtype = tf.float32)
            init = tf.global_variables_initializer()
            update_A = tf.assign(A,tf.random.uniform([n,k]))
            update_Y = tf.assign(A,tf.random.uniform([n,k]))
            #initialize session
            with tf.Session() as sess:
                sess.run(init)
                
                for option in options:
                    print_subtitle("option {}".format(option))
                    phis_label = model if option=="matrix" else model+"_"+option
                    Phi, grad_Phi, hess_Phi = predefined_Phis[phis_label]    
                    
                    times = []
                    for i in range(nb_iter):
                        sess.run(update_A)
                        start_time = time.time()
                        sess.run([Phi(A,Y),grad_Phi(A,Y),hess_Phi(A,Y)])
                        end_time = time.time() - start_time
                        times.append(end_time)
                    df_ = pd.DataFrame({'option' : [option]*nb_iter,
                                    'time' : times,
                                    'n': [n]*nb_iter})
                    df = df.append(df_, sort=False)

        with open(result_filename, "w", encoding = 'utf-8') as f:
            df.to_csv(result_filename)

    if os.path.isfile(result_filename):
        df = pd.read_csv(result_filename)
        
        sns.set()
        sns.boxplot(x="option", y="time",data=df.loc[df["n"]==1000])
        handle_fig("phi_matrix_0")
        
        sns.set()
        sns.boxplot(x="option", y="time",data=df.loc[df["n"]==10000])
        handle_fig("phi_matrix_1")
    else:
        raise FileNotFoundError
