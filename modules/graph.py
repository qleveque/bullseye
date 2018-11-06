"""
    The ``graph`` module
    ======================
 
    Contains all functions related to the creation of the Bullseye tf.graph.
 
    :Example:
 
    >>> graph, operations = construct_bullseye_graph(G)
"""

import tensorflow as tf
import numpy as np
import re
from sampling import *
from graph_related import *

"""
GRAPH
"""
def construct_bullseye_graph(G):
    """
    START GRAPH
    """
    tf.reset_default_graph()
    graph = tf.get_default_graph()
    
    """
    OPTIONS
    """
    d,k,p = [G.d, G.k, G.p]
    Phi, grad_Phi, hess_Phi, Projs = [G.Phi, G.grad_Phi, G.hess_Phi, G.Projs]
    dim_samp = p if G.local_std_trick else k #for sampling
        
    """
    X,Y
    """
    if G.X is None and G.Y is None:
        X = tf.placeholder(tf.float32, name='X', shape = [None, d])   
        Y = tf.placeholder(tf.float32, name='Y', shape = [None, k])
    else:
        X = tf.get_variable("X", G.X.shape,   initializer = tf.initializers.constant(G.X), dtype = tf.float32)
        Y = tf.get_variable("Y", G.Y.shape,  initializer = tf.initializers.constant(G.Y), dtype = tf.float32)
    
    if G.keep_1d_prior:
        prior_std = tf.get_variable("prior_std",  [p], initializer = tf.initializers.constant(np.diag(G.prior_std)), dtype = tf.float32)
    elif not G.sparse:
        prior_std = tf.get_variable("prior_std",  G.prior_std.shape, initializer = tf.initializers.constant(G.prior_std), dtype = tf.float32)
    else:
        prior_std = tf.SparseTensor(indices = [[i,i] for i in range(p)], values = [G.prior_std]*p, dense_shape = [p,p])
    
    
    """
    μ, Σ, ELBO
    """
    mu = tf.get_variable("mu",   [p],   initializer = tf.initializers.constant(G.mu_0), dtype = tf.float32)
    cov  = tf.get_variable("cov",  [p,p], initializer = tf.initializers.constant(G.cov_0), dtype = tf.float32)
    ELBO = tf.get_variable("elbo", [], initializer = tf.initializers.constant(-np.infty), dtype = tf.float32)

    """
    e, ρ, β
    """
    e = tf.get_variable("e",    [1],   initializer = tf.zeros_initializer, dtype = tf.float32)
    rho = tf.get_variable("rho",    [p],   initializer = tf.zeros_initializer, dtype = tf.float32)
    beta = tf.get_variable("beta",  [p,p], initializer = tf.initializers.constant(np.linalg.inv(G.cov_0)), dtype = tf.float32)

    """
    STEP_SIZE
    """
    step_size = tf.get_variable("step_size", [],initializer = tf.initializers.constant(G.speed), dtype = tf.float32)

    """
    new_μ, new_Σ
    """
    new_cov = tf.linalg.inv(step_size * beta + (1-step_size) * tf.linalg.inv(cov))
    new_mu  = mu - step_size * tf.einsum('ij,j->i', new_cov, rho)
    
    """
    SVD decomposition of new_Σ
    """
    new_cov_S, new_cov_U, new_cov_V = tf.linalg.svd(new_cov)
    new_cov_sqrt = tf.matmul(new_cov_U, tf.matmul(tf.linalg.diag(tf.sqrt(new_cov_S)), new_cov_V, adjoint_b=True)) #[p,p]

    """
    SAMPLING
    """
    z, z_weights = generate_normal_sampling(G.s, dim_samp)
    z_array_prior, weights_array_prior = np.polynomial.hermite.hermgauss(G.quadrature_deg)
    
    z_prior  = tf.get_variable("z_prior",  [G.quadrature_deg], initializer = tf.initializers.constant(z_array_prior), dtype = tf.float32)
    z_weights_prior  = tf.get_variable("z_weights_prior",  [G.quadrature_deg], initializer = tf.initializers.constant(weights_array_prior), dtype = tf.float32)
    """
    BATCHING : global_e, global_ρ, global_β | DEPRECATED
    """
    if G.file is not None: #streaming through a file  
        if G.chunk_as_sum:
            global_e = tf.get_variable("global_e",    [1],   initializer = tf.zeros_initializer, dtype = tf.float32)
            global_rho = tf.get_variable("global_rho",    [p],   initializer = tf.zeros_initializer, dtype = tf.float32)
            global_beta = tf.get_variable("global_beta",  [p,p], initializer = tf.initializers.constant(np.linalg.inv(G.cov_0)), dtype = tf.float32)
        else: #chunk as list
            global_e = tf.placeholder(tf.float32, name='global_e', shape = [1])
            global_rho = tf.placeholder(tf.float32, name='global_rho', shape = [p])
            global_beta = tf.placeholder(tf.float32, name='global_beta', shape = [p,p])
    else:
        global_e, global_rho, global_beta = 3*[tf.no_op()]
    """
    COMPUTATIONS OF TRIPLETS : TODO LOOK AGAIN
    """
    
    ltargs = [new_mu,new_cov,new_cov_sqrt,z,z_weights]
    #LIKELIHOOD
    if not G.m>0:
        computed_e, computed_rho, computed_beta = \
            likelihood_triplet(G,X,Y,*ltargs)
    else :
        computed_e, computed_rho, computed_beta = \
            batched_likelihood_triplet(G,X,Y,*ltargs)
            
    ptargs = [new_mu,new_cov,z_prior,z_weights_prior]
    #PRIOR
    if not G.m_prior>0:
        computed_e_prior, computed_rho_prior, computed_beta_prior =\
            prior_triplet(G, prior_std, *ptargs)
    else:
        computed_e_prior, computed_rho_prior, computed_beta_prior = batched_prior_triplet(G, prior_std, *ptargs)
    """
    FOR BATCHING
    """
    if G.file is not None and G.chunk_as_sum:
        update_global_e = tf.assign(global_e, global_e + computed_e + computed_e_prior)
        update_global_rho = tf.assign(global_rho, global_rho + computed_rho + computed_rho_prior)
        update_global_beta = tf.assign(global_beta, global_beta + computed_beta + computed_beta_prior)
        
        update_globals = [update_global_e, update_global_rho, update_global_beta]
        init_globals = tf.variables_initializer([global_e, global_rho, global_beta])
    else :
        update_globals, init_globals = [tf.no_op(), tf.no_op()]
        
    """
    NEW PARAMS ¶
    """
    if G.file is None:
        new_e = computed_e + computed_e_prior
        new_rho  = computed_rho + computed_rho_prior
        new_beta = computed_beta + computed_beta_prior
    else:
        new_e = global_e
        new_rho = global_rho
        new_beta = global_beta
    new_ELBO = - tf.squeeze(new_e) + 0.5 * tf.linalg.logdet(new_cov) + 0.5 * np.log( 2 * np.pi * np.e ) #TOSEE
    
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
    update_ops = [update_e, update_rho, update_beta, update_cov, update_mu, update_ELBO, update_step_size]
    
    """
    REFUSED UPDATE
    """
    decrease_step_size = tf.assign(step_size, step_size * G.step_size_decrease_coef, name = "decrease_step_size")
    
    """
    INIT
    """
    init = tf.global_variables_initializer()

    """
    RETURN
    """
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