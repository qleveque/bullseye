"""
    The ``graph_aux`` module
    ======================

    auxilliary module containing useful functions of the ``graph`` module.
    this module is only intended to make the code more flexible
"""

import tensorflow as tf
import numpy as np
import re

from .sampling import *

"""
ACTIVATED FUNCTIONS
"""
def activated_functions_flatten(A,Y,Phi,grad_Phi,hess_Phi):
    """
        /!\ DEPRECATED, does not work /!\
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
    phi = tf.map_fn(lambda x: Phi(x, Y), A, dtype=tf.float32) #[s,n]
    grad_phi = tf.map_fn(lambda x: grad_Phi(x, Y),A,dtype=tf.float32) #[s,n,k]
    hess_phi = tf.map_fn(lambda x: hess_Phi(x, Y), A, dtype=tf.float32) #[s,n,k,k]
    return phi, grad_phi, hess_phi


"""
COMPUTE LOCAL COV
"""

def compute_local_std_lazy(A_array, new_cov, array_is_kernel):
    #compute local_cov
    if array_is_kernel: #A_array is a kernel : A_array_kernel
        local_cov = tf.einsum('npkql,pq->nkl', A_array, new_cov) #[n,k,k]
    else:
        local_cov = tf.einsum('npk,pq,nql->nkl', A_array, new_cov, A_array) #[n,k,k]

    #compute the square_root : local_std
    #TODO to see again, is this true ?
    s_cov, u_cov, v_cov = tf.linalg.svd(local_cov)
    s_cov_sqrt = tf.linalg.diag(tf.sqrt(s_cov))
    local_std = tf.matmul(u_cov, tf.matmul(s_cov_sqrt, v_cov, adjoint_b=True))
    return local_std

def compute_local_std_trick(A_array, new_cov_sqrt):
    local_std = tf.einsum('pq,nqk->npk', new_cov_sqrt, A_array)
    return local_std

"""
TRIPLETS
"""
def likelihood_triplet(G,X,Y,new_mu,new_cov,new_cov_sqrt,z,z_weights):
    #if X.shape.as_list()[0] == 0 or Y.shape.as_list()[0] ==0:
    if tf.shape(X)[0] == 0 or tf.shape(Y)[0]==0:
        return tf.zeros([]), tf.zeros([G.p]), tf.zeros([G.p,G.p])

    A_array =  G.Projs(X, G.d, G.k) #[n,p,k]
    if G.compute_kernel:
        A_array_kernel = tf.einsum('npk,nqj->npkqj', A_array, A_array,
                                    name = "As_kernel")

    local_mu = tf.einsum('iba,b->ia',A_array, new_mu) #[n,k]

    if G.local_std_trick:
        local_std = compute_local_std_trick(A_array, new_cov_sqrt) #[n,p,k]
    else:
        A_to_use = A_array_kernel if G.compute_kernel else A_array
        local_std = compute_local_std_lazy(A_to_use, new_cov, G.compute_kernel) #[n,k,k]

    activations = tf.expand_dims(local_mu,0)\
                + tf.einsum('npk,sp->snk', local_std,z) #[s,n,k]

    activated_functions = activated_functions_flatten if G.flatten_activations\
                            else activated_functions_mapfn
    phi, grad_phi, hess_phi\
                = activated_functions(activations,Y,G.Phi,G.grad_Phi,G.hess_Phi)

    local_e = tf.einsum('s,sn->n',z_weights, phi) #[n] = [s]×[s,n]
    local_r = tf.einsum('s,snk->nk',z_weights, grad_phi) #[n,k] = [s]×[s,n,k]
    local_B = tf.einsum('s,snkj->nkj',z_weights, hess_phi) #[n,k,k] = [s]×[s,n,k,k]

    computed_e_l = tf.reduce_sum(local_e, name="computed_e_l") #[]
    computed_rho_l  = tf.einsum('npk,nk->p', A_array, local_r) #[p]=[n,p,k]×[n,k]

    if G.compute_kernel:
        #[p,p]=[n,p,k,p,k]×[n,p,k]
        computed_beta = tf.einsum('npkqj,nkj->pq',A_array_kernel, local_B)
    else:
        #computed_beta = tf.einsum('ijk,ikl,iml->jm', A_array, local_B, A_array)
        computed_beta_aux = tf.einsum('ijk,ikl->ijl', A_array, local_B)
        computed_beta = tf.einsum('ijl,iml->jm',computed_beta_aux, A_array)

    if not G.natural_param_likelihood:
        computed_e,computed_rho = \
            delocalize(computed_e_l, computed_rho_l, computed_beta, new_mu)

    else:
        computed_e, computed_rho = [computed_rho_l,comptued_e_l]
    return computed_e, computed_rho, computed_beta

def prior_triplet(G,prior_std,new_mu,new_cov,z_prior,z_weights_prior):
    if tf.shape(prior_std)[0] == 0:
        return tf.zeros([]), tf.zeros([G.p]), tf.zeros([G.p,G.p])

    if not G.keep_1d_prior:
        assert prior_std.shape.as_list()[1] == G.p
        multipliers = tf.where(tf.cast(prior_std,bool),
                               prior_std/tf.ones_like(prior_std),
                               tf.zeros_like(prior_std))
        local_mu_prior = tf.einsum('ij,j->i', multipliers, new_mu)
        if G.compute_prior_kernel:
            hess_multipliers = tf.multiply(tf.expand_dims(multipliers, 2),
                                           tf.expand_dims(multipliers, 1))
            #can take sqrt because it is of len 1
            local_std_prior = tf.sqrt(tf.einsum('ijk,jk->i',
                                                hess_multipliers,
                                                new_cov)) # [p]
        else:
            local_std_prior = tf.sqrt(tf.einsum('ij,jk,ik->i',
                                                multipliers,
                                                new_cov,
                                                multipliers))

    else:
        multipliers = 1/prior_std
        hess_multipliers = tf.square(multipliers)
        local_mu_prior = tf.einsum('i,i->i', multipliers, new_mu)
        local_std_prior = tf.sqrt(tf.einsum('i,i->i',
                                            hess_multipliers,
                                            tf.diag_part(new_cov)))

    # activations : [quadrature_deg, p]
    activations_prior = tf.expand_dims(local_mu_prior,0)\
                        + tf.expand_dims(local_std_prior,0)*tf.expand_dims(z_prior, 1)
    phi_prior = tf.square(activations_prior) / 2 # [quadrature_deg, p]
    phi_grad_prior = activations_prior # [quadrature_deg, p]
    phi_hessian_prior = tf.ones_like(activations_prior)    # [quadrature_deg, p]

    mean_phi_prior = tf.einsum('i,ij->j', z_weights_prior, phi_prior)
    mean_phi_grad_prior = tf.einsum('i,ij->j', z_weights_prior, phi_grad_prior)
    mean_phi_hessian_prior= tf.einsum('i,ij->j', z_weights_prior, phi_hessian_prior)

    computed_e_prior_l = tf.reduce_sum(mean_phi_prior)
    if not G.keep_1d_prior:
        computed_rho_prior_l  = tf.einsum('i,ij->j', mean_phi_grad_prior,multipliers)
        if G.compute_prior_kernel:
            computed_beta_prior = tf.einsum('i,ijk->jk',
                                           mean_phi_hessian_prior,
                                           hess_multipliers)
        else:
            computed_beta_prior = tf.einsum('ij,i,ik->jk',
                                            multipliers,
                                            mean_phi_hessian_prior,
                                            multipliers)
    else:
        computed_rho_prior_l  = tf.einsum('i,i->i', mean_phi_grad_prior,    multipliers)
        computed_beta_prior = tf.diag(tf.einsum('i,i->i',
                                                mean_phi_hessian_prior,
                                                hess_multipliers))

    if not G.natural_param_prior:
        computed_e_prior, computed_rho_prior = \
            delocalize(computed_e_prior_l, computed_rho_prior_l,
                        computed_beta_prior, new_mu)
    else:
        computed_rho_prior, computed_e_prior = [computed_rho_prior_l,comptued_e_prior_l]
    return computed_e_prior, computed_rho_prior, computed_beta_prior

def delocalize(e_l, rho_l, beta, mu):
    rho = rho_l - tf.einsum('i,ij->j',mu,beta)
    e = e_l - tf.einsum('i,i->',rho_l,mu) \
        + 0.5 * tf.einsum('i,ij,j->',mu, beta,mu)
    return e, rho

"""
BATCHED TRIPLETS
"""
def batched_likelihood_triplet(G,X,Y,*pars):
    x_shape = tf.shape(X)
    y_shape = tf.shape(Y)
    n = x_shape[0]
    r = n%G.m

    X_d, X_r = tf.split(X, [n-r, r], 0)
    Y_d, Y_r = tf.split(Y, [n-r, r], 0)

    X_ = tf.reshape(X_d , [tf.cast(n/G.m, tf.int32), G.m, G.d])
    Y_ = tf.reshape(Y_d , [tf.cast(n/G.m, tf.int32), G.m, G.k])

    Computed_e, Computed_rho, Computed_beta =\
        tf.map_fn(lambda x :\
            likelihood_triplet(G,x[0],x[1],*pars),\
            (X_,Y_), dtype=(tf.float32, tf.float32, tf.float32))

    computed_e_remaining, computed_rho_remaining, computed_beta_remaining =\
        likelihood_triplet(G, X_r, Y_r, *pars)

    computed_e = tf.reduce_sum(Computed_e, 0) + computed_e_remaining
    computed_rho = tf.reduce_sum(Computed_rho, 0) + computed_rho_remaining
    computed_beta = tf.reduce_sum(Computed_beta, 0) + computed_beta_remaining

    return computed_e, computed_rho, computed_beta

def batched_prior_triplet(G,prior_std,*pars):
    prior_shape = prior_std.shape.as_list()
    assert prior_shape[0] == p
    r_prior = p%G.m_prior

    prior_std_d, prior_std_r = tf.split(prior_std, [p-r_prior, r_prior], 0)
    prior_std_ = tf.reshape(prior_std_d, [tf.cast(p/G.m_prior, tf.int32), G.m_prior, p])

    Computed_e_prior, Computed_rho_prior, Computed_beta_prior =\
        tf.map_fn(lambda x : prior_triplet(G,x,*pars),
                  prior_std_,
                  dtype=(tf.float32, tf.float32, tf.float32))

    computed_e_prior_rem, computed_rho_prior_rem, computed_beta_prior_rem =\
        prior_triplet(G,prior_std_r,new_mu,new_cov)

    computed_e_prior = tf.reduce_sum(Computed_e_prior, 0)+computed_e_prior_rem
    computed_rho_prior = tf.reduce_sum(Computed_rho_prior, 0)+computed_rho_prior_rem
    computed_beta_prior = tf.reduce_sum(Computed_beta_prior, 0)+computed_beta_prior_rem

    return computed_e_prior, computd_rho_prior, computed_beta_prior
