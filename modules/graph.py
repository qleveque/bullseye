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
from argparse import Namespace
from sampling import *

"""
ACTIVATED FUNCTIONS
"""
def activated_functions_flatten(A,Y,Phi,grad_Phi,hess_Phi):
    """
        TODO : does not work properly
    """
    s,_,k = A.get_shape().as_list()
    n = tf.shape(A)[1]
    #flatten data
    A_flat = tf.reshape(A, [s*n,k])
    Y_flat = tf.tile(Y, (s,1))
    #compute flatten activated functions
    phi_flat = Phi(A_flat, Y_flat) #[s*n]
    grad_phi_flat = grad_Phi(A_flat, Y_flat) #[s*n,k]
    hess_phi_flat = hess_Phi(A_flat, Y_flat) #[s*n,k,k]
    #split to return
    phi = tf.reshape(phi_flat, [s,n])
    grad_phi = tf.reshape(grad_phi_flat, [s,n,k])
    hess_phi = tf.reshape(hess_phi_flat, [s,n,k,k])
    return phi, grad_phi, hess_phi
    
def activated_functions_mapfn(A,Y,Phi,grad_Phi,hess_Phi):
    phi = tf.map_fn(lambda x: Phi(x, Y), A, dtype=tf.float32, name="phi") #[s,n]
    grad_phi = tf.map_fn(lambda x: grad_Phi(x, Y), A, dtype=tf.float32, name = "grad_phi") #[s,n,k]
    hess_phi = tf.map_fn(lambda x: hess_Phi(x, Y), A, dtype=tf.float32, name = "hess_phi") #[s,n,k,k]
    return phi, grad_phi, hess_phi


"""
COMPUTE LOCAL COV
"""
    
def compute_local_std_lazy(A_array, new_cov, array_is_kernel):
    #compute local_cov
    if array_is_kernel: #A_array is a kernel : A_array_kernel
        local_cov = tf.einsum('npkql,pq->nkl', A_array, new_cov, name = "local_cov_square") #[n,k,k]
    else:
        local_cov = tf.einsum('npk,pq,nql->nkl', A_array, new_cov, A_array , name = "local_cov_square") #[n,k,k]
    
    #compute the square_root : local_std
    s_cov, u_cov, v_cov = tf.linalg.svd(local_cov)
    local_std = tf.matmul(u_cov, tf.matmul(tf.linalg.diag(tf.sqrt(s_cov)), v_cov, adjoint_b=True), name = "local_cov") #TOSEE^(¹/₂)
    return local_std
    
def compute_local_std_trick(A_array, new_cov_sqrt):
    local_std = tf.einsum('pq,nqk->npk', new_cov_sqrt, A_array , name = "local_cov_square")
    return local_std

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
    activated_functions = activated_functions_flatten if G.flatten_activations else activated_functions_mapfn
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
    
    if not G.sparse:
        prior_std = tf.get_variable("prior_std",  G.prior_std.shape, initializer = tf.initializers.constant(G.prior_std), dtype = tf.float32)
    else:
        #prior_std = tf.get_variable("prior_std",  G.prior_std.shape, initializer = tf.initializers.constant(G.prior_std), dtype = tf.float32)
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
    e = tf.get_variable("e",    [],   initializer = tf.zeros_initializer, dtype = tf.float32)
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
    global_e = tf.get_variable("global_e",    [],   initializer = tf.zeros_initializer, dtype = tf.float32)
    global_rho = tf.get_variable("global_rho",    [p],   initializer = tf.zeros_initializer, dtype = tf.float32)
    global_beta = tf.get_variable("global_beta",  [p,p], initializer = tf.initializers.constant(np.linalg.inv(G.cov_0)), dtype = tf.float32)
    """
    TRIPLETS
    """
    def likelihood_triplet(X,Y):
        #if X.shape.as_list()[0] == 0 or Y.shape.as_list()[0] ==0:
        if tf.shape(X)[0] == 0 or tf.shape(Y)[0]==0:
            return tf.zeros([]), tf.zeros([p]), tf.zeros([p,p])
        
        A_array =  Projs(X, d, k) #[n,p,k]
        if G.compute_kernel:
            A_array_kernel = tf.einsum('npk,nqj->npkqj',A_array,A_array, name = "As_kernel")
        
        local_mu = tf.einsum('iba,b->ia',A_array, new_mu) #[n,k]
        
        if G.local_std_trick:
            local_std = compute_local_std_trick(A_array, new_cov_sqrt) #[n,p,k]
        else:
            A_to_use = A_array_kernel if G.compute_kernel else A_array
            local_std = compute_local_std_lazy(A_to_use, new_cov, compute_kernel) #[n,k,k]
            
        activations = tf.expand_dims(local_mu,0) + tf.einsum('npk,sp->snk', local_std,z, name="activations") #[s,n,k]

        phi, grad_phi, hess_phi = activated_functions(activations,Y,Phi,grad_Phi,hess_Phi)

        local_e = tf.einsum('s,sn->n',z_weights, phi, name = "local_e") #[n] = [s]×[s,n]
        local_r = tf.einsum('s,snk->nk',z_weights, grad_phi, name="local_r") #[n,k] = [s]×[s,n,k]
        local_B = tf.einsum('s,snkj->nkj',z_weights, hess_phi, name="local_B") #[n,k,k] = [s]×[s,n,k,k]
        
        computed_e = tf.reduce_sum(local_e, name="computed_e") #[]
        computed_rho  = tf.einsum('npk,nk->p', A_array, local_r,name="computed_rho") #[p]=[n,p,k]×[n,k]
       
        if G.compute_kernel:
            computed_beta = tf.einsum('npkqj,nkj->pq', A_array_kernel, local_B, name="computed_beta") #[p,p]=[n,p,k,p,k]×[n,p,k]
        else:
            G.computed_beta = tf.einsum('ijk,ikl,iml->jm', A_array, local_B, A_array)
        
        return computed_e, computed_rho, computed_beta
    def prior_triplet(prior_std):
        #if prior_std.shape.as_list()[0] == 0 :
        if tf.shape(prior_std)[0] == 0:
            return tf.zeros([]), tf.zeros([p]), tf.zeros([p,p])
        assert prior_std.shape.as_list()[1] == p
        
        #TODO test
        #multipliers = tf.where(tf.cast(prior_std,bool), prior_std/tf.ones_like(prior_std), tf.zeros_like(prior_std))
        
        multipliers = prior_std
        
        if G.compute_prior_kernel:
            hess_multipliers = tf.multiply(tf.expand_dims(multipliers, 2), tf.expand_dims(multipliers, 1))
        local_mu_prior = tf.einsum('ij,j->i', multipliers, new_mu)
        
        #TODO add prior_std_trick
        if G.compute_prior_kernel:
            local_std_prior = tf.sqrt(tf.einsum('ijk,jk->i', hess_multipliers, cov)) # [p]
        else:
            local_std_prior = tf.sqrt(tf.einsum('ij,jk,ik->i', multipliers, cov, multipliers))
        
        # activations : [quadrature_deg, p]
        activations_prior = tf.expand_dims(local_mu_prior,0) +  tf.expand_dims(local_std_prior,0) * tf.expand_dims(z_prior, 1)
        phi_prior = tf.square(activations_prior) / 2 # [quadrature_deg, p]
        phi_grad_prior = activations_prior # [quadrature_deg, p]
        phi_hessian_prior = tf.ones_like(activations_prior)    # [quadrature_deg, p]

        mean_phi_prior = tf.einsum('i,ij->j', z_weights_prior, phi_prior)
        mean_phi_grad_prior = tf.einsum('i,ij->j', z_weights_prior, phi_grad_prior)
        mean_phi_hessian_prior= tf.einsum('i,ij->j', z_weights_prior, phi_hessian_prior)

        computed_e_prior = tf.reduce_sum(mean_phi_prior)
        computed_rho_prior  = tf.einsum('i,ij->j',   mean_phi_grad_prior,    multipliers)
        
        if G.compute_prior_kernel:
            computed_beta_prior = tf.einsum('i,ijk->jk', mean_phi_hessian_prior, hess_multipliers)
        else:
            computed_beta_prior = tf.einsum('ij,i,ik->jk', multipliers, mean_phi_hessian_prior, multipliers)
            
        return computed_e_prior, computed_rho_prior, computed_beta_prior
    
    """
    COMPUTATIONS OF TRIPLETS : TODO LOOK AGAIN
    """
    #LIKELIHOOD
    if not G.m>0:
        computed_e, computed_rho, computed_beta = likelihood_triplet(X,Y)
    else :
        x_shape = tf.shape(X)
        y_shape = tf.shape(Y)
        n = x_shape[0]
        r = n%G.m
        
        X_d, X_r = tf.split(X, [n-r, r], 0)
        Y_d, Y_r = tf.split(Y, [n-r, r], 0)
        
        X_ = tf.reshape(X_d , [tf.cast(n/G.m, tf.int32), G.m, d])
        Y_ = tf.reshape(Y_d , [tf.cast(n/G.m, tf.int32), G.m, k])
        
        Computed_e, Computed_rho, Computed_beta =\
            tf.map_fn(lambda x : likelihood_triplet(x[0],x[1]), (X_,Y_), dtype=(tf.float32, tf.float32, tf.float32))
        
        computed_e_remaining, computed_rho_remaining, computed_beta_remaining =\
            likelihood_triplet(X_r, Y_r)

        computed_e = tf.reduce_sum(Computed_e, 0) + computed_e_remaining
        computed_rho = tf.reduce_sum(Computed_rho, 0) + computed_rho_remaining
        computed_beta = tf.reduce_sum(Computed_beta, 0) + computed_beta_remaining
    #PRIOR
    if not G.m_prior>0:
        computed_e_prior, computed_rho_prior, computed_beta_prior = prior_triplet(prior_std)
    else:
        prior_shape = prior_std.shape.as_list()
        assert prior_shape[0] == p
        r_prior = p%G.m_prior
        
        prior_std_d, prior_std_r = tf.split(prior_std, [p-r_prior, r_prior], 0)
        prior_std_ = tf.reshape(prior_std_d, [tf.cast(p/G.m_prior, tf.int32), G.m_prior, p])
    
        Computed_e_prior, Computed_rho_prior, Computed_beta_prior =\
            tf.map_fn(lambda x : prior_triplet(x), prior_std_, dtype=(tf.float32, tf.float32, tf.float32))
            
        computed_e_prior_remaining, computed_rho_prior_remaining, computed_beta_prior_remaining =\
            prior_triplet(prior_std_r)
       
        computed_e_prior = tf.reduce_sum(Computed_e_prior, 0) + computed_e_prior_remaining
        computed_rho_prior = tf.reduce_sum(Computed_rho_prior, 0) + computed_rho_prior_remaining
        computed_beta_prior = tf.reduce_sum(Computed_beta_prior, 0) + computed_beta_prior_remaining
    """
    FOR BATCHING
    """
    update_global_e = tf.assign(global_e, global_e + computed_e + computed_e_prior)
    update_global_rho = tf.assign(global_rho, global_rho + computed_rho + computed_rho_prior)
    update_global_beta = tf.assign(global_beta, global_beta + computed_beta + computed_beta_prior)
    
    update_globals = [update_global_e, update_global_rho, update_global_beta]
    init_globals = tf.variables_initializer([global_e, global_rho, global_beta])
    """
    NEW PARAMS
    """
    if G.chunksize is None:
        new_e = computed_e + computed_e_prior
        new_rho  = computed_rho + computed_rho_prior
        new_beta = computed_beta + computed_beta_prior
    
    else:
        new_e = global_e
        new_rho = global_rho
        new_beta = global_beta
        
    
    new_ELBO = - new_e + 0.5 * tf.linalg.logdet(new_cov) + 0.5 * np.log( 2 * np.pi * np.e ) #TOSEE
    
    """
    ACCEPTED UPDATE
    """
    update_cov  = tf.assign(cov, new_cov, name = "update_cov")
    update_mu = tf.assign(mu, new_mu, name = "update_mu")
    update_ELBO = tf.assign(ELBO, new_ELBO, name = "update_ELBO")
    
    update_rho  = tf.assign(rho,  new_rho, name = "update_rho")
    update_beta = tf.assign(beta, new_beta, name = "update_beta")
    update_step_size = tf.assign(step_size, G.speed, name = "update_step_size")

    update_ops = [update_cov, update_mu, update_ELBO, update_rho, update_beta, update_step_size]
    
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
                'computed_e_prior' : computed_e_prior,
                'update_globals' : update_globals,
                'init_globals' : init_globals,
                'global_e' : global_e,
                'computed_e' : computed_e,
                'computed_e_prior' : computed_e_prior
                }
    
    return graph, ops_dict