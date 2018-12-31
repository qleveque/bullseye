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
result_filename = os.path.join(cwd,"data","chol_svd.data")

def std_chol_svd(recompute = False):
    if recompute:
        df = pd.DataFrame(columns=["time","method","delta","delta_inv"])
        method_names = ["cholesky","svd"]
        p = 500
        nb_iter = 300
        
        
        for method in method_names:
            print_title(method)
            
            times = []
            deltas = []
            deltas_inv = []
            
            i = 0
            while i < nb_iter: 
                i+=1
                
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
                diff_inv_chol = tf.linalg.inv(cov) - chol_inv
                diff_inv_svd = tf.linalg.inv(cov) - svd_inv
                
                init = tf.global_variables_initializer()
                
                #START COMPUTATIONS
                with tf.Session() as sess:
                    sess.run(init)
                        
                    #SVD
                    if method == "svd":
                        start_time = time.time()
                        
                        sess.run(update_svd) #eigen
                        sess.run(svd_inv) #inv
                        sess.run(svd_sqrt) #sqrt
                        sess.run(svd_logdet) # determinant
                        
                        end_time = time.time()-start_time
                    
                        delta = sess.run(tf.norm(diff_svd))
                        delta_inv = sess.run(tf.norm(diff_inv_svd))
                    
                        deltas.append(delta)
                        deltas_inv.append(delta_inv)
                        times.append(end_time)
                        print('{}\t{}\t{}'.format(end_time,delta,delta_inv))
                    
                    #Cholesky
                    elif method == "cholesky":
                        try:
                            start_time = time.time()
                            
                            sess.run(update_chol) #sqrt
                            sess.run(chol_inv) #inv
                            sess.run(chol_logdet) #determinant
                            
                            end_time = time.time()-start_time
                            
                            delta = sess.run(tf.norm(diff_chol))
                            delta_inv = sess.run(tf.norm(diff_inv_chol))
                            
                            deltas.append(delta)
                            deltas_inv.append(delta_inv)
                            times.append(end_time)
                            print('{}\t{}\t{}'.format(end_time,delta,delta_inv))
                        except:
                            print("cholesky did not work, trying it again...")
                            i-=1

            df_ = pd.DataFrame({'time' : times, 'method' : [method]*nb_iter,
                                'delta' : deltas, 'delta_inv' : deltas_inv})
            df = df.append(df_, sort=False)

        with open(result_filename, "w", encoding = 'utf-8') as f:
            df.to_csv(result_filename)

    if os.path.isfile(result_filename):
        df = pd.read_csv(result_filename)
        sns.boxplot(x="method", y="time",data=df)
        plt.title('Time required for the two different approaches.')
        handle_fig("chol_svd_times")
        
        fig = sns.boxplot(x="method", y="delta",data=df)
        fig.set_yscale('log')
        plt.title('S precisions')
        handle_fig("chol_svd_delta")
        
        fig = sns.boxplot(x="method", y="delta_inv", data=df)
        fig.set_yscale('log')
        plt.title('Σ⁻¹ precisions')
        handle_fig("chol_svd_delta_inv")
    else:
        raise FileNotFoundError
