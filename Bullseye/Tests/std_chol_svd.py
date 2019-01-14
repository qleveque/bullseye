import Bullseye
import tensorflow as tf
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import time
import pandas as pd
from tensorflow.initializers import constant as tic
from .utils import *
from Bullseye.visual import *
import math

cwd = os.path.dirname(os.path.realpath(__file__))
result_filename = os.path.join(cwd,"data","std_chol_svd.data")

def std_chol_svd(recompute = False):
    ps = [10**i for i in range(1,4)]
    if recompute:
        df = pd.DataFrame(columns=["time","method","delta"])
        method_names = ["cholesky","svd"]
        nb_iter = 100
        
        for p in ps:
            print_title("p = {}".format(p))
            #CONSTRUCT GRAPH
            tf.reset_default_graph()
            A = np.random.rand(p,p)/2.8
            d = np.linalg.det(A)
            epsilon = 0.1
            np_cov = np.dot(A,np.transpose(A)) + epsilon * np.eye(p)
            cov = tf.get_variable("cov",[p,p],
                    initializer = tic(np_cov),
                    dtype = tf.float32)

            cov_chol = tf.get_variable("cov_chol",[p,p],
                        initializer = tf.zeros_initializer,
                        dtype = tf.float32)
            
            U = tf.get_variable("U",[p,p],
                        initializer = tf.zeros_initializer,
                        dtype = tf.float32)
            S = tf.get_variable("S",[p],
                        initializer = tf.zeros_initializer,
                        dtype = tf.float32)
            V = tf.get_variable("V",[p,p],
                        initializer = tf.zeros_initializer,
                        dtype = tf.float32)
            
            cov_chol_ = tf.linalg.cholesky(cov)
            S_, U_, V_ = tf.linalg.svd(cov)
            
            update_chol = tf.assign(cov_chol,cov_chol_)
            update_svd = [tf.assign(U,U_),
                        tf.assign(S,S_),
                        tf.assign(V,V_)]
            
            chol_inv = tf.cholesky_solve(cov_chol, tf.eye(p))
            chol_logdet = tf.reduce_sum(tf.log(tf.linalg.diag_part(cov_chol)))
            
            svd_inv = U @ tf.matmul(tf.linalg.diag(tf.reciprocal(S)),
                                    V, adjoint_b=True)
            svd_sqrt = U @ tf.matmul(tf.linalg.diag(tf.sqrt(S)),
                                                V, adjoint_b=True)
            svd_logdet = tf.reduce_sum(tf.log(S))
            
            diff_chol = cov - cov_chol @ tf.transpose(cov_chol)
            diff_svd = cov - svd_sqrt @ tf.transpose(svd_sqrt)
            
            init = tf.global_variables_initializer()
            
            #START COMPUTATIONS
            with tf.Session() as sess:
                sess.run(init)
                
                for method in method_names:
                    print_subtitle(method)
                    times = []
                    deltas = []
                    
                    for i in range(nb_iter):
                        start_time = time.time()
                        #SVD
                        if method == "svd":
                            sess.run(update_svd)
                            sess.run(svd_sqrt) #sqrt
                            sess.run(svd_logdet) # determinant
                            sess.run(svd_inv) #inv
                            
                            end_time = time.time()-start_time
                        
                            delta = sess.run(tf.norm(diff_svd))
                        
                            deltas.append(delta)
                        
                        #Cholesky
                        elif method == "cholesky":
                            sess.run(update_chol) #sqrt
                            sess.run(chol_inv) #inv
                            sess.run(chol_logdet) #determinant
                            
                            end_time = time.time()-start_time
                            
                            delta = sess.run(tf.norm(diff_chol))
                            
                            deltas.append(delta)
                        
                        times.append(end_time)

                    df_ = pd.DataFrame({'time' : times, 'method' : [method]*nb_iter,
                                        'delta' : deltas, 'p' : [p]*nb_iter})
                    df = df.append(df_, sort=False)

        with open(result_filename, "w", encoding = 'utf-8') as f:
            df.to_csv(result_filename)

    if os.path.isfile(result_filename):
        df = pd.read_csv(result_filename)
        for i in range(len(ps)):
            sns.set()
            sns.boxplot(x="method", y="time",data=df.loc[df['p']==ps[i]],
                        showfliers = False)
            handle_fig("chol_svd_times_{}".format(i))
        
        df_ = df.groupby(['p','method']).mean()["delta"]
        print(df_)
        print(df_.to_latex())
    else:
        raise FileNotFoundError
