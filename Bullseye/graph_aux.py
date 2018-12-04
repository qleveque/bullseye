"""
    The ``graph_aux`` module
    ========================

    Auxilliary module containing useful functions for the ``graph`` module.
    This module is only intended to make the code more flexible
"""

import tensorflow as tf
import numpy as np
import re

from .sampling import *

"""
TRIPLETS
"""

def likelihood_triplet(G,X,Y,new_mu,new_cov,new_cov_sqrt,z,z_weights):
    """
    Describes the part of the tensorflow graph related to the computation of e,
    ρ and β.

    Parameters
    ----------
    G : Bullseye.Graph
        The considered bullseye graph object.
    X : tf.tensor [n,d]
        The design matrix.
    Y : tf.tensor [n,k]
        The response matrix.
    new_mu : tf.tensor [p]
        The new μ candidate that is being studied.
    new_cov : tf.tensor [p,p]
        The new Σ candidate that is being studied.
    new_cov_sqrt : tf.tensor [p,p]
        The square root of the new Σ candidate that is being studied.
    z : tf.tensor [p] (or [k] if not G.local_std_trick)
        A sample of the standardized normal law
    z_weights :
        The weights of each observation of the sample.

    Returns
    -------
    e : tf.tensor[]
        Computed e for given X,Y.
    rho : tf.tensor[p]
        Computed ρ for given X,Y.
    beta : tf.tensor[p,p]
        Computed β for given X,Y.

    """
    pars = [G,X,Y,new_mu,new_cov,new_cov_sqrt,z,z_weights]
    if G.use_projs :
        return soft_likelihood_triplet(*pars)
    else:
        return brutal_likelihood_triplet(*pars)

def prior_triplet(G,prior_std,new_mu,new_cov,new_cov_sqrt,
                  z_prior,z_weights_prior):
    """
    Describes the part of the tensorflow graph related to the computation of e,
    ρ and β for the prior.

    Parameters
    ----------
    G : Bullseye.Graph
        The considered bullseye graph object.
    std_prior: tf.tensor [p,p] (or [p] if G.keep_1d_prior)
        The normal prior standard deviation.
    new_mu : tf.tensor [p]
        The new μ candidate that is being studied.
    new_cov : tf.tensor [p,p]
        The new Σ candidate that is being studied.
    new_cov_sqrt : tf.tensor [p,p]
        The square root of the new Σ candidate that is being studied.
    z : tf.tensor [p] (or [k] if not G.local_std_trick)
        A sample of the standardized normal law
    z_weights :
        The weights of each observation of the sample.

    Returns
    -------
    e_prior : tf.tensor[]
        Computed e of the prior for given X,Y.
    rho_prior : tf.tensor[p]
        Computed ρ for the prior given X,Y.
    beta_prior : tf.tensor[p,p]
        Computed β for the prior given X,Y.

    """
    pars = [G,prior_std,new_mu,new_cov,new_cov_sqrt,z_prior,z_weights_prior]
    if G.use_projs :
        return soft_prior_triplet(*pars)
    else:
        return brutal_prior_triplet(*pars)

"""
OVERLOADS
"""

def brutal_likelihood_triplet(G,X,Y,new_mu,new_cov,new_cov_sqrt,z,z_weights):
    """
    Overload function of ``likelihood_triplet``.
    Is called when we don't consider the projections of the parameters in the
    bullseye algorithm.
    """
    if tf.shape(X)[0] == 0 or tf.shape(Y)[0]==0:
        return tf.zeros([]), tf.zeros([G.p]), tf.zeros([G.p,G.p])


    #in this case, we have to use the std_trick
    assert G.local_std_trick

    #from sample z, compute the corresponding activations :
    # Activations[j] = aⱼ = μ+σ·zⱼ           of size [s,n,k]
    Activations = tf.expand_dims(new_mu,0)\
                + tf.einsum('pk,sp->sk', new_cov_sqrt,z,
                            name = 'einsum_in_activations')

    #activate the function with the computed activations:
    #compute:
    # psi[j]=ψ(aⱼ)                   of size [s]
    # grad_psi[j]=∇ψ(aⱼ)             of size [s,k]
    # hess_psi[j]=Hψ(aⱼ)             of size [s,k,k]
    psi = tf.map_fn(lambda theta: G.Psi(X, Y, theta), Activations,
                        dtype=tf.float32)
    grad_psi = tf.map_fn(lambda theta: G.grad_Psi(X, Y, theta), Activations,
                        dtype=tf.float32)
    hess_psi = tf.map_fn(lambda theta: G.hess_Psi(X, Y, theta), Activations,
                        dtype=tf.float32)

    #compute the real parameters:
    #computed_e = e* = ∑ⱼ wⱼ·ψ(aⱼ) ≈ 𝔼[ψ(aⱼ)]              of size []
    #computed_rho = ρ* = ∑ⱼ wⱼ·∇ψ(aⱼ) ≈ 𝔼[∇ψ(aⱼ)]          of size [k]
    #computed_beta = β = ∑ⱼ wⱼ·Hψ(aⱼ) ≈ 𝔼[Hψ(aⱼ)]          of size [k,k]
    computed_e = tf.einsum('s,s->',z_weights, psi)
    computed_rho = tf.einsum('s,sk->k',z_weights, grad_psi)
    computed_beta = tf.einsum('s,skj->kj',z_weights, hess_psi)

    #→
    if not G.natural_param_likelihood:
        #in case we don't want to be centered around 0
        computed_e_del, computed_rho_del = \
            delocalize(computed_e, computed_rho, computed_beta, new_mu)
        #and return e, ρ and β
        return computed_e_del, computed_rho_del, computed_beta

    #otherwise, return e*, ρ* and β
    return computed_e, computed_rho, computed_beta

def soft_likelihood_triplet(G,X,Y,new_mu,new_cov,new_cov_sqrt,z,z_weights):
    if tf.shape(X)[0] == 0 or tf.shape(Y)[0]==0:
        return tf.zeros([]), tf.zeros([G.p]), tf.zeros([G.p,G.p])
    """
    Overload function of ``likelihood_triplet``.
    Is called when we consider the projections of the parameters in the bullseye
    algorithm.
    """
    #consider i, ∀i ∈〚1,n〛

    #compute projection arrays:
    # A_array[i] = Aᵢ
    # A_array_kernel[i] = [A^T·A]ᵢ
    #note that A_array_kernel may not be computed, depending on the options of G
    A_array, A_array_kernel = aux_A_arrays(G,X)

    #compute local parameters, in other terms describe how behaves Aᵢ·θ:
    # local_mu[i]=μᵢ
    # local_std[i]=σᵢ
    local_mu, local_std = aux_local_parameters(G, A_array, A_array_kernel,
                                               new_mu, new_cov, new_cov_sqrt)

    #from sample z, compute the corresponding activations :
    # Activations[j,i] = aᵢ = μᵢ+σᵢzⱼ       of size [s,n,k]
    Activations = tf.expand_dims(local_mu,0) +\
                  tf.einsum('npk,sp->snk', local_std,z,
                            name = 'einsum_in_activations')

    #activate the functions with the computed activations
    #compute :
    # phi[j,i]=ϕᵢ(Aᵢzⱼ),            of size [s,n]
    # grad_phi[j,i]=∇ϕᵢ(Aᵢzⱼ)       of size [s,n,k]
    # hess_phi[j,i]=Hϕᵢ(Aᵢzⱼ)       of size [s,n,k,k]
    phi, grad_phi, hess_phi = aux_activate_functions(G,Activations,Y)

    #compute the parameters e, r and B :
    # local_e[i] = ∑ⱼ wⱼ·ϕᵢ(Aᵢzⱼ) ≈ 𝔼[ϕᵢ(Aᵢθ)] = eᵢ      of size [n]
    # local_r[i] = ∑ⱼ wⱼ·∇ϕᵢ(Aᵢzⱼ) ≈ 𝔼[∇ϕᵢ(Aᵢθ)] = rᵢ    of size [n,k]
    # local_B[i] = ∑ⱼ wⱼ·Hϕᵢ(Aᵢzⱼ) ≈ 𝔼[Hϕᵢ(Aᵢθ)] = Bᵢ    of size [n,k,k]
    local_e = tf.einsum('s,sn->n',z_weights, phi)
    local_r = tf.einsum('s,snk->nk',z_weights, grad_phi)
    local_B = tf.einsum('s,snkj->nkj',z_weights, hess_phi)

    #finally compute the real parameters :
    # computed_e = e* = ∑ᵢ eᵢ                           of size []
    # computed_rho = ρ* = ∑ᵢ Aᵢ·rᵢ = ∑ᵢ ρᵢ              of size [p]
    # computed_beta = β = ∑ᵢ (Aᵢ^T)·Bᵢ·Aᵢ = ∑ᵢ βᵢ       of size[p,p]
    computed_e = tf.reduce_sum(local_e, name="computed_e_l")
    computed_rho  = tf.einsum('npk,nk->p', A_array, local_r,
                              name = 'computed_rho_l')
    computed_beta = aux_compute_beta(G, A_array_kernel, A_array, local_B)
    #→
    if not G.natural_param_likelihood:
        #in case we don't want to be centered around 0
        computed_e_del, computed_rho_del = \
            delocalize(computed_e, computed_rho, computed_beta, new_mu)
        #and return e, ρ and β
        return computed_e_del, computed_rho_del, computed_beta

    #otherwise, return e*, ρ* and β
    return computed_e, computed_rho, computed_beta

def brutal_prior_triplet(G,prior_std,new_mu,new_cov,new_cov_sqrt,
                        z_prior,z_weights_prior):
    """
    WILL CHANGE IN FUTURE VERSIONS →
    so i don't take the time to comment, code it in another way
    """
    if tf.shape(prior_std)[0] == 0:
        return tf.zeros([]), tf.zeros([G.p]), tf.zeros([G.p,G.p])

    assert G.keep_1d_prior

    multipliers = 1/prior_std
    hess_multipliers = tf.square(multipliers)
    local_mu_prior = tf.einsum('i,i->i', multipliers, new_mu)
    local_std_prior = tf.sqrt(tf.einsum('i,i->i',
                                        hess_multipliers,
                                        tf.diag_part(new_cov)))

    # activations : [quadrature_deg, p]
    activations_prior = tf.expand_dims(local_mu_prior,0)\
                  + tf.expand_dims(local_std_prior,0)*tf.expand_dims(z_prior, 1)

    # [quadrature_deg, p]
    phi_prior = tf.square(activations_prior) / 2
    # [quadrature_deg, p]
    phi_grad_prior = activations_prior
    # [quadrature_deg, p]
    phi_hessian_prior = tf.ones_like(activations_prior)

    mean_phi_prior = tf.einsum('i,ij->j', z_weights_prior, phi_prior)
    mean_phi_grad_prior = tf.einsum('i,ij->j', z_weights_prior, phi_grad_prior)
    mean_phi_hessian_prior= tf.einsum('i,ij->j', z_weights_prior,
                                                 phi_hessian_prior)

    computed_e_prior_l = tf.reduce_sum(mean_phi_prior)
    computed_rho_prior_l  = mean_phi_grad_prior
    computed_beta_prior = tf.diag(mean_phi_hessian_prior)

    if not G.natural_param_prior:
        computed_e_prior, computed_rho_prior = \
            delocalize(computed_e_prior_l, computed_rho_prior_l,
                        computed_beta_prior, new_mu)
    else:
        computed_rho_prior = computed_rho_prior_l
        computed_e_prior = computed_e_prior_l
    return computed_e_prior, computed_rho_prior, computed_beta_prior



def soft_prior_triplet(G,prior_std,new_mu,new_cov,new_cov_sqrt,
                        z_prior,z_weights_prior):
    """
    WILL CHANGE IN FUTURE VERSIONS →
    so i don't take the time to comment, code it in another way
    """
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
    # [quadrature_deg, p]
    phi_prior = tf.square(activations_prior) / 2
    # [quadrature_deg, p]
    phi_grad_prior = activations_prior
    # [quadrature_deg, p]
    phi_hessian_prior = tf.ones_like(activations_prior)

    mean_phi_prior = tf.einsum('i,ij->j', z_weights_prior, phi_prior)
    mean_phi_grad_prior = tf.einsum('i,ij->j', z_weights_prior, phi_grad_prior)
    mean_phi_hessian_prior= tf.einsum('i,ij->j', z_weights_prior,
                                                 phi_hessian_prior)

    computed_e_prior_l = tf.reduce_sum(mean_phi_prior)
    if not G.keep_1d_prior:
        computed_rho_prior_l  = tf.einsum('i,ij->j', mean_phi_grad_prior,
                                                     multipliers)
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
        computed_rho_prior_l  = tf.einsum('i,i->i', mean_phi_grad_prior,
                                                    multipliers)
        computed_beta_prior = tf.diag(tf.einsum('i,i->i',
                                                mean_phi_hessian_prior,
                                                hess_multipliers))

    if not G.natural_param_prior:
        computed_e_prior, computed_rho_prior = \
            delocalize(computed_e_prior_l, computed_rho_prior_l,
                        computed_beta_prior, new_mu)
    else:
        computed_rho_prior = computed_rho_prior_l
        computed_e_prior = computed_e_prior_l
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
    """
    Overlay function of ``likelihood_triplet``.
    Is called when we want to apply ``likelihood_triplet`` on X and Y using
    batches.
    """
    #retrieve the current shapes
    x_shape = tf.shape(X)
    y_shape = tf.shape(Y)
    #the number of observations
    n = x_shape[0]
    #each batch will contain m elements, but the last one, which will contain
    #r elements
    r = n%G.m

    #remove the rest from the input matrices
    X_d, X_r = tf.split(X, [n-r, r], 0)
    Y_d, Y_r = tf.split(Y, [n-r, r], 0)

    #reshape what remains into the different batches
    X_ = tf.reshape(X_d , [tf.cast(n/G.m, tf.int32), G.m, G.d])
    Y_ = tf.reshape(Y_d , [tf.cast(n/G.m, tf.int32), G.m, G.k])

    #compute e, rho and beta for these batches
    Computed_e, Computed_rho, Computed_beta =\
        tf.map_fn(lambda x :\
            likelihood_triplet(G,x[0],x[1],*pars),\
            (X_,Y_), dtype=(tf.float32, tf.float32, tf.float32))

    #compute e, rho and beta for the rest matrix
    computed_e_remaining, computed_rho_remaining, computed_beta_remaining =\
        likelihood_triplet(G, X_r, Y_r, *pars)

    #compute the sum of all those computed parameters e, rho and beta
    computed_e = tf.reduce_sum(Computed_e, 0) + computed_e_remaining
    computed_rho = tf.reduce_sum(Computed_rho, 0) + computed_rho_remaining
    computed_beta = tf.reduce_sum(Computed_beta, 0) + computed_beta_remaining

    return computed_e, computed_rho, computed_beta

def batched_prior_triplet(G,prior_std,*pars):
    """
    Overlay function of ``prior_triplet``.
    Is called when we want to apply ``prior_triplet`` on prior_std using
    batches.
    """
    #retrieve the current shapes
    prior_shape = prior_std.shape.as_list()
    r_prior = p%G.m_prior

    #check the sizes
    assert prior_shape[0] == p

    #remove the rest from the input matrices
    prior_std_d, prior_std_r = tf.split(prior_std, [p-r_prior, r_prior], 0)

    #reshape what remains into the different batches
    prior_std_ = tf.reshape(prior_std_d,
                            [tf.cast(p/G.m_prior, tf.int32), G.m_prior, p])

    #compute e, rho and beta for these batches
    Computed_e_prior, Computed_rho_prior, Computed_beta_prior =\
        tf.map_fn(lambda x : prior_triplet(G,x,*pars),
                  prior_std_,
                  dtype=(tf.float32, tf.float32, tf.float32))

    #compute e, rho and beta for the rest matrix
    computed_e_prior_rem, computed_rho_prior_rem, computed_beta_prior_rem =\
        prior_triplet(G,prior_std_r,new_mu,new_cov)


    #compute the sum of all those computed parameters e, rho and beta
    computed_e_prior = tf.reduce_sum(Computed_e_prior, 0) +\
                        computed_e_prior_rem
    computed_rho_prior = tf.reduce_sum(Computed_rho_prior, 0) +\
                        computed_rho_prior_rem
    computed_beta_prior = tf.reduce_sum(Computed_beta_prior, 0) +\
                        computed_beta_prior_rem

    return computed_e_prior, computd_rho_prior, computed_beta_prior

"""
AUXILLIARY FUNCTIONS
"""

def aux_A_arrays(G, X):
    """
    Make the computation of the different projection matrices Aᵢ.


    Returns
    -------
    A_array : tf.tensor [n,p,k]
        The set of projection matrices where A_array[i]=Aᵢ
    A_array_kernel : tf.tensor [n,p,k,p,k], optional
        The set of projection matrix kernels, where
        A_array_kernel[i,a,b,c,d] = Aᵢ[a,b] × Aᵢ[c,d]
    """
    #A_array[i]=Aᵢ              of size [n,p,k]
    A_array =  G.Proj(X, G.d, G.k)

    #compute optionally the kernel
    A_array_kernel = None
    if G.compute_kernel:
        A_array_kernel = tf.einsum('npk,nqj->npkqj', A_array, A_array,
                                    name = "As_kernel")

    return A_array, A_array_kernel

def aux_local_parameters(G,A_array,A_array_kernel,new_mu,new_cov,new_cov_sqrt):
    """
    Make the computation of the different local parameters, μᵢ and σᵢ.

    Returns
    -------
    local_mu : tf.tensor [n,k]
        𝔼[Aᵢ·θ]
    local_std : tf.tensor [n,k,k]
        sd[Aᵢ·θ]=√Var[Aᵢ·θ]
    """
    #compute local_mu:
    # local_mu[i] = μᵢ = Aᵢ·μ               of size [k]
    local_mu = tf.einsum('iba,b->ia',A_array, new_mu,
                          name = 'einsum_local_mu') #[n,k]

    #compute local_std
    # local_std[i] = σᵢ = √Var[Aᵢ·θ]
    if G.local_std_trick:
        #local std trick, using new_cov_sqrt, no use of A_array_kernel

        #compute:
        # local_std = σᵢ = √Σ·Aᵢ        of size [n,p,k]
        if G.use_einsum:
            local_std = tf.einsum('pq,nqk->npk', new_cov_sqrt, A_array,
                                   name = 'einsum_std_trick')
        else:
            local_std = tf.transpose(
                                    tf.tensordot(new_cov_sqrt,
                                                 A_array,[[1],[1]],
                                                 name = "tensordot_std_trick"),
                                    [1,0,2])
    else:
        #in this case, no std trick, so we compute local_cov, and then local_std
        # local_std[i] = √local_cov[i] = √(Aᵢ·Σ·(Aᵢ^T))
        if G.compute_kernel:
            local_cov = tf.einsum('npkql,pq->nkl', A_array_kernel, new_cov,
                                 name = 'einsum_lazy_kernel_local_cov')
        else:
            local_cov = tf.einsum('npk,pq,nql->nkl', A_array, new_cov, A_array,
                                 name = 'einsum_lazy_local_cov')
        local_std = matrix_sqrt(local_cov)

    return local_mu, local_std

def aux_activate_psi_functions(G, Activations, Y):
    """
    →→→
    Compute the activated functions ψ(a), ∇ψ(a), Hψ(a), ∀a ∈ Activations.
    """
    #using flatten Activations
    if G.flatten_activations:
        s,_,k = A.get_shape().as_list()
        n = tf.shape(A)[1]
        #flatten data
        Activations_flat = tf.reshape(Activations, [s*n,k])
        Y_flat = tf.tile(Y, (s,1))

        #compute flatten activated functions
        #recall we have Activations of size [s×n]
        #compute:
        # phi[j]=ψ(aⱼ)                          of size [s×n]
        # grad_phi[j]=∇ψ(aⱼ)                    of size [s×n,k]
        # hess_phi[j]=Hψ(aⱼ)                    of size [s×n,k,k]
        psi_flat = G.Psi(Activations_flat, Y_flat) #[s*n]
        grad_psi_flat = G.grad_Psi(Activations_flat, Y_flat) #[s*n,k]
        hess_psi_flat = G.hess_Psi(Activations_flat, Y_flat) #[s*n,k,k]

        #unflatten data
        psi = tf.reshape(phi_flat, [s,n])
        grad_psi = tf.reshape(grad_phi_flat, [s,n,k])
        hess_psi = tf.reshape(hess_phi_flat, [s,n,k,k])

    #using map_fn
    else:
        #recall we have Activations of size [s,n].
        #compute:
        # phi[j,i]=ψ(aⱼᵢ)                       of size [s,n]
        # grad_phi[j,i]=∇ψ(aⱼᵢ)                 of size [s,n,k]
        # hess_phi[j,i]=Hψ(aⱼᵢ)                 of size [s,n,k,k]
        psi = tf.map_fn(lambda a: G.Psi(a, Y), Activations,
            dtype=tf.float32)
        grad_psi = tf.map_fn(lambda a: G.grad_Psi(a, Y), Activations,
            dtype=tf.float32)
        hess_psi = tf.map_fn(lambda a: G.hess_Psi(a, Y), Activations,
            dtype=tf.float32)

    return phi, grad_phi, hess_phi

def aux_activate_functions(G, Activations, Y):
    """
    Compute the activated functions ϕ(a), ∇ϕ(a), Hϕ(a), ∀a ∈ Activations.
    """
    #using flatten Activations
    if G.flatten_activations:
        s,_,k = A.get_shape().as_list()
        n = tf.shape(A)[1]
        #flatten data
        Activations_flat = tf.reshape(Activations, [s*n,k])
        Y_flat = tf.tile(Y, (s,1))

        #compute flatten activated functions
        #recall we have Activations of size [s×n]
        #compute:
        # phi[j]=ϕ(aⱼ)                          of size [s×n]
        # grad_phi[j]=∇ϕ(aⱼ)                    of size [s×n,k]
        # hess_phi[j]=Hϕ(aⱼ)                    of size [s×n,k,k]
        phi_flat = G.Phi(Activations_flat, Y_flat) #[s*n]
        grad_phi_flat = G.grad_Phi(Activations_flat, Y_flat) #[s*n,k]
        hess_phi_flat = G.hess_Phi(Activations_flat, Y_flat) #[s*n,k,k]

        #unflatten data
        phi = tf.reshape(phi_flat, [s,n])
        grad_phi = tf.reshape(grad_phi_flat, [s,n,k])
        hess_phi = tf.reshape(hess_phi_flat, [s,n,k,k])

    #using map_fn
    else:
        #recall we have Activations of size [s,n].
        #compute:
        # phi[j,i]=ϕ(aⱼᵢ)                       of size [s,n]
        # grad_phi[j,i]=∇ϕ(aⱼᵢ)                 of size [s,n,k]
        # hess_phi[j,i]=Hϕ(aⱼᵢ)                 of size [s,n,k,k]
        phi = tf.map_fn(lambda a: G.Phi(a, Y), Activations,
            dtype=tf.float32)
        grad_phi = tf.map_fn(lambda a: G.grad_Phi(a, Y), Activations,
            dtype=tf.float32)
        hess_phi = tf.map_fn(lambda a: G.hess_Phi(a, Y), Activations,
            dtype=tf.float32)

    return phi, grad_phi, hess_phi

def aux_compute_beta(G, A_array_kernel, A_array, local_B):
    """
    Compute β from B in the case we are using projected parameters.
    """
    #compute β
    # computed_beta = β = ∑ᵢ (Aᵢ^T)·Bᵢ·Aᵢ = ∑ᵢ βᵢ       of size [p,p]
    if G.compute_kernel:
        computed_beta = tf.einsum('npkqj,nkj->pq',A_array_kernel, local_B,
                                  name = 'einsum_likelihood_compute_kernel')
    else:
        computed_beta_ = tf.einsum('ijk,ikl->ijl', A_array, local_B,
                                      name = 'computed_beta_aux')
        computed_beta = tf.einsum('ijl,iml->jm',computed_beta_, A_array,
                                 name = 'computed_beta')
    return computed_beta

"""
AUTO GRAD HESS
"""

def auto_grad_Psi(Psi,X,Y,theta):
    #→
    return tf.gradients(Psi(X,Y,theta),theta)[0]

def auto_hess_Psi(Psi,X,Y,theta):
    #→
    J = auto_grad_Psi(Psi,X,Y,theta)
    return hess_from_grad(J)

def auto_grad_Phi(Phi,A,Y):
    #→
    return tf.gradients(Phi(A,Y),theta)[0]

def auto_hess_Phi(Phi,A,Y):
    #→
    J = auto_grad_Phi(Psi,X,Y,theta)
    return hess_from_grad(J)

def hess_from_grad(grad):
    return tf.tensordot(grad,grad,axes=0)
