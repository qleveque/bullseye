"""
    The ``Bullseye`` module
    ======================
 
    Contains the definition of the Bullseye_graph class, allowing to create a graph with specific options and to launch the Bullseye algorithm.
    
    :Example:
    >>> from Bullseye import Bullseye_graph
    >>> bull = Bullseye_graph()
    >>> bull.feed_with(file = "dataset.csv", chunksize = 50, k=10)
    >>> bull.set_model("multilogit")
    >>> bull.init_with(mu_0 = 0, cov_0 = 1)
    >>> bull.set_options(m = 0, compute_kernel = True)
    >>> bull.build()
    >>> bull.run()
"""

from graph import construct_bullseye_graph
from predefined_functions import get_predefined_functions
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import time
from utils import *

class Bullseye_graph:
    def __init__(self):
        attrs = ["graph",
            #data
            "d","k","p","X","Y","prior_std","file","chunksize",
            #init
            "mu_0","cov_0",
            #functions
            "Phi","grad_Phi","hess_Phi","Projs"
            ]
            
        options = {"speed"                  : 1,
                "step_size_decrease_coef"   : 0.5,
                "flatten_activations"       : False,
                "local_std_trick"           : True ,
                "m"                         : 0,
                "m_prior"                   : 0,
                "compute_kernel"            : False,
                "compute_prior_kernel"      : False,
                "s"                         : 10,
                "quadrature_deg"            : 12,
                "sparse"                    : True,
                "chunk_as_sum"              : True,
                "keep_1d_prior"             : True,
                "number_of_chunk_max"       : 0,
                "natural_param_prior"       : False,
                "natural_param_likelihood"  : False
                }        
        self.option_list = list(options)
        
        self.in_graph = None
        
        for key in attrs:
            setattr(self, key, None)
        for key in self.option_list:
            setattr(self, key, options[key])
            
        self.feed_with_is_called = False
        self.set_model_is_called = False
        self.init_with_is_called = False
        self.build_is_called = False
        
    def feed_with(self, X = None, Y = None, d = None, k = None, prior_std = None, file = None, chunksize = None, **kwargs):
    
        if X is not None or Y is not None:
            assert X is not None
            assert Y is not None
            
            assert Y.shape[0] == X.shape[0]
            assert len(X.shape) in [1,2]
            assert len(Y.shape) in [1,2]
            
            if len(X.shape)==1:
                X = expand_dims(X,0)
            if len(Y.shape)==1:
                Y = expand_dims(Y,0)
            
            self.X = X
            self.Y = Y
            self.d = X.shape[-1]
            self.k = Y.shape[-1]
        elif file is not None:
            #TODO to see again
            assert os.path.isfile(file)
            if chunksize is not None:
                assert type(chunksize)==int
                
            assert k is not None
            assert type(k) == int
            
            reader = pd.read_table(file, sep=",", chunksize = 1)
            for chunk in reader:
                data = list(chunk.shape)
                break
            
            self.k = k
            self.d = data[-1]-1
            
            self.file = file
            self.chunksize = chunksize
        else:
            assert d is not None
            assert k is not None
            assert type(d), type(k) == [int,int]            
            
            self.d = d
            self.k = k
            
        self.p = self.d * self.k
       
        if prior_std is None:
            self.prior_std = np.eye(self.p)
        elif type(prior_std) == int:
            self.prior_std = prior_std * np.eye(self.p)
        elif len(list(prior_std.shape)) == 1:
            assert list(prior_std.shape)==[self.p]
            self.prior_std = np.diag(prior_std)
        else:
            assert list(prior_std.shape) == [self.p, self.p]
            self.prior_std = prior_std
        
        self.feed_with_is_called = True
            
    
    def set_model(self, model = None, Phi = None, grad_Phi = None, hess_Phi = None, Projs = None, **kwargs):
        if model is not None:
            self.Phi, self.grad_Phi, self.hess_Phi, self.Projs = get_predefined_functions(model, **kwargs)
        else:
            assert Phi is not None
            self.Phi = Phi
            self.grad_Phi = grad_Phi
            self.hess_Phi = hess_Phi
            self.Projs = Projs
            
        self.set_model_is_called = True
    
    def init_with(self, mu_0 = None, cov_0 = None):
        if mu_0 == None:
            self.mu_0 = np.zeros(self.p)
        elif type(mu_0) in [float, int]:
            self.mu_0 = mu_0 * np.ones(self.p)
        elif len(mu_0.shape) == 1:
            assert list(mu_0.shape) == [p]
            self.mu_0 = mu_0
        
        if cov_0 == None:
            self.cov_0 = np.eye(self.p)
        elif type(cov_0) in [float, int]:
            self.cov_0 = cov_0 * np.eye(self.p)
        elif len(cov_0.shape) == 1:
            assert list(mu_0.shape) == [self.p]
            self.cov_0 = np.diag(cov_0)
        else:
            assert list(mu_0.shape) == [self.p,self.p]
            self.cov_0 = cov_0
        self.init_with_is_called = True
            
    def set_options(self, **kwargs):
        for key in list(kwargs):
            if key not in self.option_list:
                raise ValueError('Unknown parameter given for set_options().\nSpecifically "{}".'.format(key))
            else:
                setattr(self, key, kwargs[key])
                
        #TODO : compute_prior_kernel useless when using keep_1d_prior
        #WARNING
        if self.keep_1d_prior:
            self.compute_prior_kernel = False
    
    def build(self):
        assert self.feed_with_is_called and self.set_model_is_called and self.init_with_is_called
        
        self.graph, self.in_graph = construct_bullseye_graph(self)
        self.build_is_called = True
    
    def run(self, n_iter = 10, show_elapsed_time = True, dict = {}):
        assert self.build_is_called
        
        ops = self.in_graph
        mus = []
        covs = []
        elbos = []
        elbo = -np.inf
        
        with tf.Session(graph = self.graph) as sess:
            #INIT
            sess.run(ops["init"])
            
            #START
            if show_elapsed_time:
                start_time = time.time()
            
            for epoch in range(n_iter):
                if self.file is None:
                    new_elbo  = sess.run(ops["new_ELBO"], feed_dict = dict)
                    d_computed = {}
                else:
                    sess.run(ops["init_globals"])
                    reader = pd.read_table(self.file, sep = ",", chunksize = self.chunksize)
                    d_computed = self.read_chunks(reader,sess)
                                  
                new_elbo  = sess.run(ops["new_ELBO"], feed_dict = d_computed)
                if new_elbo<=elbo:
                    sess.run(ops["decrease_step_size"])
                    print("not accepted : {}".format(new_elbo))
                else:
                    sess.run(ops["update_ops"], feed_dict = d_computed)
                    elbo = new_elbo
                    print("accepted : {}".format(elbo))
                    
                mu, cov = sess.run([ops["mu"], ops["cov"]])                
                
                mus.append(mu)
                covs.append(cov)
                elbos.append(elbo)
            
            if show_elapsed_time:
                elapsed_time = time.time()-start_time
                print('\nit took {} s'.format(elapsed_time))
                
            return {"mus" : mus, "covs" : covs, "elbos" : elbos}
            
    def read_chunks(self, reader, sess):
        ops = self.in_graph
    
        if not self.chunk_as_sum:
            computed_e_ = np.empty((0,1),np.float32)
            computed_rho_ = np.empty((0,self.p),np.float32)
            computed_beta_ = np.empty((0,self.p,self.p),np.float32)
        
        i = 0
        for chunk in reader:
            data = np.asarray(chunk)
            X = data[:,1:]/253.
            Y = to_one_hot(data[:,0])
            d_ = {"X:0" : X, "Y:0" : Y}
            if self.chunk_as_sum:
                sess.run(ops["update_globals"], feed_dict = d_)
            else: #chunk as list
                to_compute = ["computed_e","computed_rho","computed_beta",
                    "computed_e_prior","computed_rho_prior","computed_beta_prior"]
                ce, cr, cb, cep, crp, cbp =\
                    sess.run([ops[op] for op in to_compute], feed_dict = d_)
                
                computed_e_ = np.vstack((computed_e_,(ce + cep)))
                computed_rho_ = np.vstack((computed_rho_,(cr + crp)))
                computed_beta_ = np.vstack((computed_beta_,np.expand_dims((cb + cbp),0)))
            print("chunk done")
            if self.number_of_chunk_max != 0:
                i+=1
                if i>=self.number_of_chunk_max:
                    break
                    
        d_computed = {}
        if not self.chunk_as_sum:
            d_computed = {"global_e:0": np.sum(computed_e_, axis = 0),
                        "global_rho:0": np.sum(computed_rho_, axis = 0),
                        "global_beta:0": np.sum(computed_beta_, axis = 0)}
        return d_computed