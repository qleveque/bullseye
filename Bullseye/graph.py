"""
    The ``graph`` module
    ======================
 
    Contains all functions related to the creation of the Bullseye tensorflow graph.
 
    :Example:
    >>> graph, operations = construct_bullseye_graph(G)
"""

import tensorflow as tf
from tensorflow.initializers import constant as tic
import numpy as np
import re

from .sampling import *
from .graph_aux import *

"""
GRAPH
"""
def construct_bullseye_graph(G):
    #set the graph
    tf.reset_default_graph()
    graph = tf.get_default_graph()
    
    #for simplicity
    d,k,p = [G.d, G.k, G.p]
    Phi, grad_Phi, hess_Phi, Projs = [G.Phi, G.grad_Phi, G.hess_Phi, G.Projs]
    dim_samp = p if G.local_std_trick else k
    
    """
    VARIABLES
    """
    
    #X, Y related
    if G.X is None and G.Y is None:
        X = tf.placeholder(tf.float32, name='X', shape = [None, d])   
        Y = tf.placeholder(tf.float32, name='Y', shape = [None, k])
    else:
        X = tf.get_variable("X", G.X.shape,initializer = tic(G.X),dtype = tf.float32)
        Y = tf.get_variable("Y",G.Y.shape,initializer = tic(G.Y),dtype = tf.float32)
    
    #prior_std related
    if G.keep_1d_prior:
        prior_std = tf.get_variable("prior_std", 
                        [p],
                        initializer = tic(np.diag(G.prior_std)),
                        dtype = tf.float32)
    elif not G.sparse:
        prior_std = tf.get_variable("prior_std", 
                        G.prior_std.shape,
                        initializer = tic(G.prior_std),
                        dtype = tf.float32)
    else:
        prior_std = tf.SparseTensor(indices = [[i,i] for i in range(p)],
                                    values = [G.prior_std]*p,
                                    dense_shape = [p,p])
    
    
    #μ, Σ and ELBO related
    mu = tf.get_variable("mu",[p],initializer = tic(G.mu_0),dtype = tf.float32)
    cov  = tf.get_variable("cov",[p,p],initializer = tic(G.cov_0),dtype = tf.float32)
    ELBO = tf.get_variable("elbo",[],initializer = tic(-np.infty),dtype = tf.float32)

    #e, ρ and β related
    e = tf.get_variable("e",[],initializer = tf.zeros_initializer,dtype = tf.float32)
    rho = tf.get_variable("rho",
                          [p],
                          initializer = tf.zeros_initializer,
                          dtype = tf.float32)
    beta = tf.get_variable("beta",
                           [p,p],
                           initializer = tic(np.linalg.inv(G.cov_0)),
                           dtype = tf.float32)

    #step size
    step_size = tf.get_variable("step_size",
                                [],
                                initializer = tic(G.speed),
                                dtype = tf.float32)

    #new_cov, new_mu
    new_cov = tf.linalg.inv(step_size * beta + (1-step_size) * tf.linalg.inv(cov))
    new_mu  = mu - step_size * tf.einsum('ij,j->i', new_cov, rho)
    
    #SVD decomposition of new_cov
    new_cov_S, new_cov_U, new_cov_V = tf.linalg.svd(new_cov)
    new_cov_S_sqrt = tf.linalg.diag(tf.sqrt(new_cov_S))
    new_cov_sqrt = tf.matmul(new_cov_U,
                             tf.matmul(new_cov_S_sqrt,new_cov_V, adjoint_b=True)) #[p,p]

    #sampling related
    z, z_weights = generate_sampling_tf(G.s, dim_samp)
    
    z_array_prior, weights_array_prior =\
        np.polynomial.hermite.hermgauss(G.quadrature_deg)
    z_prior  = tf.get_variable("z_prior",
                                [G.quadrature_deg],
                                initializer = tic(z_array_prior),
                                dtype = tf.float32)
    z_weights_prior  = tf.get_variable("z_weights_prior", 
                            [G.quadrature_deg],
                            initializer = tic(weights_array_prior),
                            dtype = tf.float32)

    """
    TRIPLETS
    """
    #LIKELIHOOD TRIPLET
    #for readability
    ltargs = [new_mu,new_cov,new_cov_sqrt,z,z_weights]
    #if not batched
    if not G.m>0:
        computed_e, computed_rho, computed_beta = \
            likelihood_triplet(G,X,Y,*ltargs)
    #if batched
    else :
        computed_e, computed_rho, computed_beta = \
            batched_likelihood_triplet(G,X,Y,*ltargs)
            
    #PRIO TRIPLET
    #for readability
    ptargs = [new_mu,new_cov,z_prior,z_weights_prior]
    #if not batched
    if not G.m_prior>0:
        computed_e_prior, computed_rho_prior, computed_beta_prior =\
            prior_triplet(G, prior_std, *ptargs)
    #if batched
    else:
        computed_e_prior, computed_rho_prior, computed_beta_prior =\
            batched_prior_triplet(G, prior_std, *ptargs)
    
    """
    FOR CHUNKS
    """
    #if streaming through a file as sum
    if G.file is not None and G.chunk_as_sum:
        global_e = tf.get_variable("global_e",
                                    [],
                                    initializer = tf.zeros_initializer,
                                    dtype = tf.float32)
        global_rho = tf.get_variable("global_rho",
                                    [p],
                                    initializer = tf.zeros_initializer,
                                    dtype = tf.float32)
        global_beta = tf.get_variable("global_beta",
                                    [p,p],
                                    initializer = tic(np.linalg.inv(G.cov_0)),
                                    dtype = tf.float32)
        
        #chunk as sum, will increase step by step global_e, global_rho and global_beta
        # with update_globals
        update_global_e = tf.assign(global_e,
                                    global_e + computed_e + computed_e_prior)
        update_global_rho = tf.assign(global_rho,
                                    global_rho + computed_rho + computed_rho_prior)
        update_global_beta = tf.assign(global_beta,
                                    global_beta + computed_beta + computed_beta_prior)
        
        update_globals = [update_global_e, update_global_rho, update_global_beta]
        init_globals = tf.variables_initializer([global_e, global_rho, global_beta])
        
    #chunk as list : the sum will be computed outside of the graph
    elif G.file is not None and not G.chunk_as_sum:
        global_e = tf.placeholder(tf.float32, name='global_e', shape = [])
        global_rho = tf.placeholder(tf.float32, name='global_rho', shape = [p])
        global_beta = tf.placeholder(tf.float32, name='global_beta', shape = [p,p])
        update_globals, init_globals = 2*[tf.no_op()]
    
    #if not streaming through a file, they will not be used
    else:
        global_e, global_rho, global_beta, update_globals, init_globals = 5*[tf.no_op()]
    
        
    """
    NEW PARAMS
    """
    #if G.file is None, we are directly using triplets results
    if G.file is None:
        new_e = computed_e + computed_e_prior
        new_rho  = computed_rho + computed_rho_prior
        new_beta = computed_beta + computed_beta_prior
        
    #if G.file, we use global_e, global_rho and global_beta
    else:
        new_e = global_e
        new_rho = global_rho
        new_beta = global_beta
        
    #new ELBO
    new_ELBO = - new_e \
               + 0.5 * tf.linalg.logdet(new_cov) \
               + 0.5 * np.log(2*np.pi*np.e)
    
    """
    ACCEPTED UPDATE
    """
    update_e = tf.assign(e, new_e, name = "update_e")
    update_rho  = tf.assign(rho,  new_rho, name = "update_rho")
    update_beta = tf.assign(beta, new_beta, name = "update_beta")
    
    update_cov  = tf.assign(cov, new_cov, name = "update_cov")
    update_mu = tf.assign(mu, new_mu, name = "update_mu")
    update_ELBO = tf.assign(ELBO, new_ELBO, name = "update_ELBO")
    
    update_step_size = tf.assign(step_size, G.speed, name = "update_step_size")
    update_ops = [update_e, update_rho, update_beta,
                  update_cov, update_mu,
                  update_ELBO, update_step_size]
    
    """
    REFUSED UPDATE
    """
    decrease_step_size = tf.assign(step_size, step_size*G.step_size_decrease_coef)
    
    """
    INIT and RETURN
    """
    init = tf.global_variables_initializer()
    ops_dict = {'init' : init,
                'new_ELBO' : new_ELBO,
                'ELBO' : ELBO,
                'update_ops' : update_ops,
                'decrease_step_size' : decrease_step_size,
                'mu' : mu,
                'cov' : cov,
                'rho' : rho,
                'beta' : beta,
                'X' : X,
                'Y' : Y,
                'update_globals' : update_globals,
                'init_globals' : init_globals,
                'global_e' : global_e,
                
                'computed_e' : computed_e,
                'computed_e_prior' : computed_e_prior,
                'computed_rho' : computed_rho,
                'computed_rho_prior' : computed_rho_prior,
                'computed_beta' : computed_beta,
                'computed_beta_prior' : computed_beta_prior
                }
    
    return graph, ops_dict