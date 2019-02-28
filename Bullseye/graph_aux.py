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

def likelihood_triplet(G,X,Y,new_mu,new_cov,new_cov_sqrt):
    """
    Describes the part of the tensorflow graph related to the computation of e,
    ρ and \beta.

    Parameters
    ----------
    G : Bullseye.Graph
        The considered bullseye graph object.
    X : tf.tensor [n,d]
        The design matrix.
    Y : tf.tensor [n,k]
        The response matrix.
    new_mu : tf.tensor [p]
        The new \mu candidate that is being studied.
    new_cov : tf.tensor [p,p]
        The new \Sigma candidate that is being studied.
    new_cov_sqrt : tf.tensor [p,p]
        The square root of the new \Sigma candidate that is being studied.

    Returns
    -------
    e : tf.tensor[]
        Computed e for given X,Y.
    rho : tf.tensor[p]
        Computed ρ for given X,Y.
    beta : tf.tensor[p,p]
        Computed \beta for given X,Y.

    """
    pars = [G,X,Y,new_mu,new_cov,new_cov_sqrt]
    
    if G.use_projs :
        return proj_likelihood_triplet(*pars)
    else:
        return brutal_likelihood_triplet(*pars)

"""
OVERLOADS
"""

def brutal_likelihood_triplet(G,X,Y,new_mu,new_cov,new_cov_sqrt):
    """
    Overload function of ``likelihood_triplet``.
    Is called when we don't consider the projections of the parameters in the
    bullseye algorithm.
    """
    if tf.shape(X)[0] == 0 or tf.shape(Y)[0]==0:
        return tf.zeros([]), tf.zeros([G.p]), tf.zeros([G.p,G.p])

    #sample s realisations of Z_i~𝒩(0,1)
    z, z_weights = generate_sampling_tf(G.s, G.p)

    #from sample z, compute the corresponding activations :
    # thetas[j] = \theta_j = \mu+σ\cdotz_j           of size [s,n,k]
    # \theta_j is a realisation of \theta~𝒩(\mu,\Sigma)
    thetas = tf.expand_dims(new_mu,0)\
                + tf.einsum('pk,sp->sk', new_cov_sqrt,z,
                            name = 'einsum_in_activations')
    
    #activate the function with the computed activations:
    #compute:
    # psi[j]=\psi(\theta_j)                   of size [s]
    # grad_psi[j]=\nabla\psi(\theta_j)             of size [s,k]
    # hess_psi[j]=H\psi(\theta_j)             of size [s,k,k]
    psi, grad_psi, hess_psi = compute_psis(G,X,Y,thetas,new_mu,new_cov)
    
    #compute the real parameters:
    #computed_e = e* = \sum_j w_j\cdot\psi(\theta_j) ≈ E[\psi(\theta_j)]              of size []
    #computed_rho = ρ* = \sum_j w_j\cdot\nabla\psi(\theta_j) ≈ E[\nabla\psi(\theta_j)]          of size [k]
    #computed_beta = \beta = \sum_j w_j\cdotH\psi(\theta_j) ≈ E[H\psi(\theta_j)]          of size [k,k]
    computed_e = tf.einsum('s,s->',z_weights, psi)
    computed_rho = tf.einsum('s,sk->k',z_weights, grad_psi)
    if not G.diag_cov:
        computed_beta = tf.einsum('s,skj->kj',z_weights, hess_psi)
    else:
        computed_beta = tf.einsum('s,sk->k',z_weights, hess_psi)
    
    return relocalize(computed_e, computed_rho, computed_beta,
                      new_mu,new_cov, G.diag_cov)

"""
def test_(G,X,Y,new_mu,new_cov,new_cov_sqrt):
    if tf.shape(X)[0] == 0 or tf.shape(Y)[0]==0:
        return tf.zeros([]), tf.zeros([G.p]), tf.zeros([G.p,G.p])

    #sample s realisations of Z_i~𝒩(0,1)
    z, z_weights = generate_sampling_tf(G.s, G.p)
    thetas = tf.expand_dims(new_mu,0)\
                + tf.einsum('pk,sp->sk', new_cov_sqrt,z,
                            name = 'einsum_in_activations')
    psi = tf.map_fn(lambda t: G.Psi(X, Y, t), thetas,
                        dtype=tf.float32)
    return psi
"""
def proj_likelihood_triplet(G,X,Y,new_mu,new_cov,new_cov_sqrt):
    if tf.shape(X)[0] == 0 or tf.shape(Y)[0]==0:
        return tf.zeros([]), tf.zeros([G.p]), tf.zeros([G.p,G.p])
    """
    Overload function of ``likelihood_triplet``.
    Is called when we consider the projections of the parameters in the bullseye
    algorithm.
    """
    #consider i, ∀i \in[1,n]

    #compute projection arrays:
    # A_array[i] = A_i               of size [p,k]
    A_array = G.Proj(X, G.k)

    #compute local parameters, in other terms describe how behaves A_i\cdot\theta:
    # local_mu[i]=\mu_i
    # local_std[i]=√\Sigma_i
    local_mu, local_std, local_cov = aux_local_parameters(G, A_array,
                                               new_mu, new_cov, new_cov_sqrt)
    
    #sample s realisations of Z_i~𝒩(0,1)
    l=G.p if not G.local_std_trick else G.k
    z, z_weights = generate_sampling_tf(G.s, l)

    #from sample z, compute the corresponding activations :
    # Activations[j,i] = A_i\theta_j = = \mu_i+√\Sigma_iz_j       of size [s,l,k]
    Activations = tf.expand_dims(local_mu,0) +\
                  tf.einsum('npk,sp->snk', local_std,z,
                            name = 'einsum_in_activations')

    #activate the functions with the computed activations
    #compute :
    ##
    # phi[j,i] = \varphi(A_i\theta_j) = \phi_i(\theta_j),                  of size [s,n]
    # grad_phi[j,i] = \nabla\varphi(A_i\theta_j) = \nabla\phi_i(\theta_j)            of size [s,n,k]
    # hess_phi[j,i] = H\varphi(A_i\theta_j) = H\phi_i(\theta_j)            of size [s,n,k,k]
    phi, grad_phi, hess_phi = \
        compute_phis(G,Activations,Y,local_mu, local_std, local_cov)
    
    #compute the parameters e, r and B :
    # local_e[i] = \sum_j w_j\cdot\varphi_i(A_iz_j) ≈ E[\varphi_i(A_i\theta)] = e_i      of size [n]
    # local_r[i] = \sum_j w_j\cdot\nabla\varphi_i(A_iz_j) ≈ E[\nabla\varphi_i(A_i\theta)] = r_i    of size [n,k]
    # local_B[i] = \sum_j w_j\cdotH\varphi_i(A_iz_j) ≈ E[H\varphi_i(A_i\theta)] = B_i    of size [n,k,k]
    local_e = tf.einsum('s,sn->n',z_weights, phi)
    local_r = tf.einsum('s,snk->nk',z_weights, grad_phi)
    local_B = tf.einsum('s,snkj->nkj',z_weights, hess_phi)

    #finally compute the real parameters :
    # computed_e = e* = \sum_i e_i                           of size []
    # computed_rho = ρ* = \sum_i A_i\cdotr_i = \sum_i ρ_i              of size [p]
    # computed_beta = \beta = \sum_i (A_i^T)\cdotB_i\cdotA_i = \sum_i \beta_i       of size[p,p]
    computed_e = tf.reduce_sum(local_e, name="computed_e_l")
    computed_rho  = tf.einsum('npk,nk->p', A_array, local_r,
                              name = 'computed_rho_l')
    computed_beta = aux_compute_beta(G, A_array, local_B)
    
    return relocalize(computed_e, computed_rho, computed_beta,
                      new_mu, new_cov, G.diag_cov)

def prior_triplet(G,new_mu,new_cov,new_cov_sqrt):
    """
    Describes the part of the tensorflow graph related to the computation of e,
    ρ and \beta for the prior.

    Parameters
    ----------
    G : Bullseye.Graph
        The considered bullseye graph object.
    new_mu : tf.tensor [p]
        The new \mu candidate that is being studied.
    new_cov : tf.tensor [p,p]
        The new \Sigma candidate that is being studied.
    new_cov_sqrt : tf.tensor [p,p]
        The square root of the new \Sigma candidate that is being studied.
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
        Computed \beta for the prior given X,Y.

    """
    l=G.p if not G.prior_iid else 1
    
    z, z_weights = generate_sampling_tf(G.s, l)


    if not G.prior_iid:
        #from sample z, compute the corresponding activations :
        # thetas[j] = \theta_j = \mu+√\Sigma\cdotz_j           of size [s_q,p]
        # \theta_j is a realisation of \theta~𝒩(\mu,\Sigma)
        thetas = tf.expand_dims(new_mu,0)\
                    + tf.einsum('pk,sp->sk', new_cov_sqrt,z)
            
    else:
        # Prior suppose the component of \theta to be independant
        # We don't need to sample \theta, but rather each component of \theta
        # thetas[i][j] = \theta_i_j = \mu_j+z_i\cdot√\Sigma_j_j           of size [s_q,p]
        # \theta_i_j is a realisation of \theta[j]~𝒩(\mu_j,\Sigma_j_j)
        new_cov_diag_sqrt = tf.sqrt(tf.linalg.diag_part(new_cov))
        thetas = tf.expand_dims(new_mu,0)\
                    + tf.einsum('p,sp->sp', new_cov_diag_sqrt,z)
    
    #recall we have Activations of size [s_q,p].
    #compute:
    # pi[j]=\pi(a_j)                       of size [s_q]
    # grad_pi[j]=\nabla\pi(a_j)                 of size [s_q,p]
    # hess_pi[j]=H\pi(a_j)                 of size [s_q,p,p]
    pi = tf.map_fn(lambda theta : G.Pi(theta), thetas,
            dtype=tf.float32)
    grad_pi = tf.map_fn(G.grad_Pi, thetas,
            dtype=tf.float32)
    
    #independant coordinates
    if not G.prior_iid:
        hess_Pi = G.hess_Pi
    else:
        hess_Pi=lambda x : tf.linalg.diag_part(G.hess_Pi(x))    
    hess_pi = tf.map_fn(hess_Pi, thetas, dtype=tf.float32)
    
    # computed_e_l[i] = \sum_j w_j\cdot\pi(a_j) ≈ E[\pi(\theta)] = e*_\pi             of size []
    # computed_rho_l[i] = \sum_j w_j\cdot\nabla\pi(A_iz_j) ≈ E[\nabla\pi(\theta)] = ρ*_\pi       of size [p]
    # computed_beta_l[i] = \sum_j w_j\cdotH\pi(A_iz_j) ≈ E[H\pi(\theta)] = \beta_\pi       of size [p,p]
    computed_e_l = tf.einsum('s,s->', z_weights, pi)
    computed_rho_l = tf.einsum('s,sk->k', z_weights, grad_pi)
    if not G.prior_iid:
        computed_beta_l = tf.einsum('s,skj->kj', z_weights, hess_pi)
    else:
        computed_beta_l = tf.einsum('s,sk->k', z_weights, hess_pi)
        
    return relocalize(computed_e_l, computed_rho_l, computed_beta_l,
                      new_mu, new_cov, G.prior_iid)


"""
AUXILLIARY FUNCTIONS
"""
def relocalize(e_l, rho_l, beta_l, mu, cov, beta_diag=False):
    #"""
    if not beta_diag:
        return e_l, rho_l, beta_l
    else:
        return e_l, rho_l, tf.diag(beta_l)
    #"""
    
    if not beta_diag:
        beta = beta_l
        mu_beta = tf.einsum('i,ij->j',mu,beta_l)
        mu_beta_mu = tf.einsum('i,ij,j->',mu, beta_l ,mu)
    else:
        beta = tf.diag(beta_l)
        mu_beta = tf.einsum('i,i->i',mu,beta_l)
        mu_beta_mu = tf.einsum('i,i->',mu_beta,mu)
    
    rho = rho_l - mu_beta
    e = e_l - tf.einsum('i,i->',rho_l,mu) + 0.5 * mu_beta_mu
    
    #-0.5 * tf.diag(tf.linalg.diag_part(tf.einsum('ij,jk->ik',beta,cov)))
    
    return e, rho, beta

def compute_phis(G, Activations, Y, local_mu, local_std, local_cov):
    """
    Compute the activated functions \varphi(a), \nabla\varphi(a), H\varphi(a), ∀a \in Activations.
    """
    s,_,k = Activations.get_shape().as_list()
    n = tf.shape(Activations)[1]
        
    #Y
    if G.flatten_activations :
        Y_ = tf.tile(Y, (s,1))
    else:
        Y_ = Y
    
    #\phi
    Phi_ = lambda a : G.Phi(a,Y_)
    
    #\nabla\phi
    grad_Phi_ = None
    if G.grad_Phi is not None:
        grad_Phi_ = lambda a: G.grad_Phi(a,Y_)
        
    #H\phi
    #impossible to compute from tf
    hess_Phi_ = None
    if G.hess_Phi is not None:
        hess_Phi_ = lambda a : G.hess_Phi(a,Y_)
        
    #if an approximation is required, we need w
    approx_needed = grad_Phi_ is None or hess_Phi_ is None
    if approx_needed:
        if local_cov is None:
            #local_cov[i] = \Sigma_i= S^T\cdotS                   of size [n,k,k]
            local_cov = tf.einsum('nji,njk->nik',local_std,local_std)
        #cov_inv[i] = \Sigma_i^{-1}                   of size [n,k,k]
        cov_inv = tf.linalg.inv(local_cov)
        #v[j,i] = A_i\theta_j-\mu_i                  of size [s,n,k]
        v = Activations - tf.expand_dims(local_mu,0)
        #w[j,i] = \Sigma_i^{-1}(A_i\theta_j-\mu_i) = \Sigma_i^{-1}\cdotv[j,i]  of size [s,n,k]
        w = tf.einsum('npq,snq->snp',cov_inv,v)
    
    #using flatten Activations
    if G.flatten_activations:
        #flatten data
        Activations_flat = tf.reshape(Activations, [s*n,k])
        if approx_needed:
            w_flat = tf.reshape(w, [s*n,k])
            cov_inv_flat = tf.tile(cov_inv,[s,1,1])
        #compute flatten activated functions
        #recall we have Activations of size [s\timesn]
        #compute:
        # phi[j]=\varphi(a_j)                          of size [s\timesn]
        phi_flat = Phi_(Activations_flat)
        
        # grad_phi[j]=\nabla\varphi(a_j)                    of size [s\timesn,k]
        if grad_Phi_ is not None:
            grad_phi_flat = grad_Phi_(Activations_flat)
        else:
            grad_phi_flat = grad_approx(G, w_flat, phi_flat)
        
        # hess_phi[j]=H\varphi(a_j)                    of size [s\timesn,k,k]
        if hess_Phi_ is not None:
            hess_phi_flat = hess_Phi_(Activations_flat)
        else:
            hess_phi_flat = hess_approx(G, w_flat, cov_inv_flat,phi_flat, grad_phi_flat)
        
        #unflatten data
        phi = tf.reshape(phi_flat, [s,n])
        grad_phi = tf.reshape(grad_phi_flat, [s,n,k])
        hess_phi = tf.reshape(hess_phi_flat, [s,n,k,k])

    #using map_fn
    else:
        #recall we have Activations of size [s,l,k].
        #compute:
        # phi[j,i]=\varphi(A_i\theta_j)                       of size [s,n]
        phi = tf.map_fn(Phi_, Activations, dtype=tf.float32)
        
        # grad_phi[j,i]=\nabla\varphi(A_i\theta_j)                 of size [s,n,k]
        if grad_Phi_ is not None:
            grad_phi = tf.map_fn(grad_Phi_, Activations, dtype=tf.float32)
        else:
            grad_phi = tf.map_fn(lambda A : grad_approx(G, A[0], A[1]), [w,phi], dtype=tf.float32)
        
        # hess_phi[j,i]=H\varphi(A_i\theta_j)                 of size [s,n,k,k]
        if hess_Phi_ is not None:
            hess_phi = tf.map_fn(hess_Phi_, Activations, dtype=tf.float32)
        else:
            hess_phi = tf.map_fn(lambda A : hess_approx(G, A[0],cov_inv,A[1],A[2]),
                                 [w,phi,grad_phi], dtype=tf.float32)
        
    return phi, grad_phi, hess_phi

def compute_psis(G, X, Y, thetas, mu, cov):
    grad_psi = None
    hess_psi = None
    
    #\psi
    psi = tf.map_fn(lambda t: G.Psi(X, Y, t), thetas,
                        dtype=tf.float32)
    #\nabla\psi
    if G.grad_Psi is not None:
        gp = lambda t: G.grad_Psi(X,Y,t)
        grad_psi = tf.map_fn(gp, thetas, dtype=tf.float32)
        
    #H\psi
    if G.hess_Psi is not None:
        if not G.diag_cov:
            hp = lambda t: G.hess_Psi(X,Y,t)
        else:
            hp = lambda t: tf.linalg.diag_part(G.hess_Psi(X,Y,t))
        hess_psi = tf.map_fn(hp, thetas, dtype=tf.float32)
        #if both are computed
        if grad_psi is not None:
            return psi, grad_psi, hess_psi
    
    #APPROXIMATIONS
    
    #cov_inv = \Sigma^{-1}                  of size [p,p]
    cov_inv = tf.linalg.inv(cov)
    #v[i] = \theta_i-\mu                  of size [s,p]
    v = thetas - tf.expand_dims(mu,0)
    #w[i] = \Sigma^{-1}(\theta_i-\mu) = \Sigma^{-1}\cdotv[i]  of size [s,p,p]
    w = tf.einsum('pq,sq->sp',cov_inv,v)
    
    if grad_psi is None:
        grad_psi = grad_approx(G,w,psi)
    if hess_psi is None :
        hess_psi = hess_approx(G,w,cov_inv,psi,grad_psi)
        
    return psi, grad_psi, hess_psi
    
def grad_approx(G,w,act):
    #using the activations previously computed
    grad = tf.einsum('n,np->np',act,w)
    return grad
    
def hess_approx(G,w,cov_inv,act,grad):
    if G.compute_hess in ["grad","tf"]:
        #using the gradient previously computed
        #Hƒ(\theta)[i] = Sym(\nablaƒ(\theta)(\theta-\mu_i)^T \Sigma_i^{-1})
        #         = Sym(\nablaƒ(\theta)\cdotw_i^T)
        if not G.diag_cov:
            outer = tf.einsum('np,nq->npq',grad,w)    
            return Sym(outer)
        else:
            outer = tf.einsum('np,np->np',grad,w)
            return outer
    
    else : #G.compute_hess == "act":
        #using the activations of f previously computed
        #Hƒ(\theta)[i] = 0.5\cdotƒ(\theta)(\Sigma^{-1}(\theta-\mu_i)(\theta-\mu_i)^T \Sigma_i^{-1} - \Sigma_i^{-1})
        #         = 0.5\cdotƒ(\theta)(w[i]\cdotw[i]^T - I)
        if not G.diag_cov:
            outer = tf.einsum('np,nq->npq', w, w)
            I = tf.expand_dims(tf.eye(G.k),0)
            inside = 0.5 * (outer - cov_inv)
            r = tf.einsum('i,ijk->ijk',act,inside)
            return r
        else:
            outer = tf.einsum('np,np->np',w,w)
            I = tf.expand_dims(tf.ones(G.k),0)
            inside = 0.5 * (outer - I)
            r = tf.einsum('i,ij->ij',act,inside)
            return r

def compute_hess(G, f, thetas, mu, cov, act, grad):
    """
    Compute the hessians of the function f given its sample thetas.
    """
    if G.compute_hess == "std":
        return tf.map_fn(f, thetas, dtype=tf.float32)
    
    #cov_inv = \Sigma^{-1}                  of size [p,p]
    #to improve
    cov_inv = tf.linalg.inv(cov)
    #v[i] = \theta[i]-\mu                  of size [s,p]
    v = thetas - tf.expand_dims(mu, 0)
    #w[i] = \Sigma^{-1}(\theta[i]-\mu) = \Sigma^{-1}\cdotv[i]  of size [s,p]
    w = tf.einsum('pq,sq->sp',cov_inv,v)
    
    if G.compute_hess == "grad":
        #using the gradient previously computed
        #Hƒ(\theta_i) = Sym(\nablaƒ(\theta)(\theta-\mu)^T \Sigma^{-1})
        #       = Sym(\nablaƒ(\theta)\cdotw_i^T)
        #of size [s,p,p]
        outer = tf.einsum('sp,sq->spq',grad,w)    
        return Sym(outer)
    
    if G.compute_hess == "act":
        #using the activations of f previously computed
        #Hƒ(\theta_i) = 0.5\cdotƒ(\theta_i)(\Sigma^{-1}(\theta_i-\mu)(\theta_i-\mu)^T \Sigma^{-1} - I)
        #       = 0.5\cdotƒ(\theta_i)(w[i]\cdotw[i]^T - I)
        #of size [s,p,p]
        outer = tf.einsum('sp,sq->spq', w, w)
        p = tf.shape(cov_inv)[0]
        I = tf.expand_dims(tf.eye(p),0)
        inside = 0.5 * (outer - I)
        return tf.einsum('i,ijk->ijk', act,inside)
    
"""
OTHER AUXILLIARY
"""
"""
def aux_A_arrays(G, X):
    DEPRECATED
    Make the computation of the different projection matrices A_i.


    Returns
    -------
    A_array : tf.tensor [n,p,k]
        The set of projection matrices where A_array[i]=A_i
    A_array_kernel : tf.tensor [n,p,k,p,k], optional
        The set of projection matrix kernels, where
        A_array_kernel[i,a,b,c,d] = A_i[a,b] \times A_i[c,d]
    
    
    #A_array[i]=A_i              of size [n,p,k]
    A_array =  G.Proj(X, G.k)

    #compute optionally the kernel
    A_array_kernel = None
    if G.compute_kernel:
        if not G.diag_cov:
            A_array_kernel = tf.einsum('npk,nqj->npkqj', A_array, A_array,
                                        name = "As_kernel")
        else:
            A_array_kernel = tf.einsum('npk,npj->npkj', A_array, A_array,
                                       name = "As_kernel")

    return A_array, A_array_kernel
"""
def aux_local_parameters(G,A_array,new_mu,new_cov,new_cov_sqrt):
    """
    Make the computation of the different local parameters, \mu_i and σ_i.

    Returns
    -------
    local_mu : tf.tensor [n,k]
        E[A_i\cdot\theta]
    local_std : tf.tensor [n,k,k]
        sd[A_i\cdot\theta]=√Var[A_i\cdot\theta]
    """
    #compute local_mu:
    # local_mu[i] = \mu_i = A_i\cdot\mu               of size [n,k]
    local_mu = tf.einsum('iba,b->ia',A_array, new_mu,
                          name = 'einsum_local_mu') #[n,k]

    #compute local_std
    # local_std[i] = σ_i = √Var[A_i\cdot\theta]
    if not G.local_std_trick:
        #using new_cov_sqrt
        local_cov = None
        #compute:
        # local_std = σ_i = √\Sigma\cdotA_i        of size [n,p,k]
        local_std = tf.einsum('pq,nqk->npk', new_cov_sqrt, A_array,
                                name = 'einsum_std_trick')
    else:
        #local_cov = tf.einsum('npk,pq,nql->nkl', A_array, new_cov, A_array,
        #                         name = 'einsum_lazy_local_cov')
        local_cov_ = tf.einsum('npk,pq->nkq',A_array,new_cov)
        local_cov = tf.einsum('nkq,nql->nkl', local_cov_, A_array)
        local_std_T = tf.linalg.cholesky(local_cov)
        local_std = tf.transpose(local_std_T, perm=[0, 2, 1])

    return local_mu, local_std, local_cov

def aux_compute_beta(G, A_array, local_B):
    """
    Compute \beta from B in the case we are using projected parameters.
    """
    #compute \beta
    # computed_beta = \beta = \sum_i (A_i^T)\cdotB_i\cdotA_i = \sum_i \beta_i       of size [p,p]
    computed_beta_ = tf.einsum('ijk,ikl->ijl', A_array, local_B,
                                    name = 'computed_beta_aux')
    if not G.diag_cov:
        computed_beta = tf.einsum('ijl,iml->jm',computed_beta_, A_array,
                                name = 'computed_beta')
    else:
        computed_beta = tf.einsum('ijl,ijl->j', computed_beta_, A_array,
                                name = 'computed_beta')
    return computed_beta

def compute_new_cov_and_co(G,gamma,cov, cov_max,
                    cov_max_inv,
                    new_cov_sqrt):
    
    new_cov_sqrt_ = None
    if G.backtracking_degree==-1:
        # \Sigma^{n+1} = (γ\cdot(\Sigmaₘ)^{-1} + (1-γ)\cdot(\Sigmaⁿ)^{-1})^{-1}
        new_cov_=tf.linalg.inv(gamma * cov_max_inv \
                               + (1-gamma) * tf.linalg.inv(cov))
    elif G.backtracking_degree==1:
        # \Sigma^{n+1} = γ\cdot\Sigmaₘ + (1-γ)\cdot\Sigmaⁿ
        new_cov_ = gamma*(cov_max) + (1-gamma) * cov
        
    elif G.backtracking_degree==0.5:
        #S^{n+1} = γ\cdot\Sigmaₘ^(^{1/2}) + (1-γ)\cdotSⁿ                with Sⁿ = (\Sigmaⁿ)^^{1/2}
        new_cov_sqrt_ = gamma*(cov_max_sqrt) \
                        + (1-gamma) * new_cov_sqrt
        new_cov_ = new_cov_sqrt_ @ tf.transpose(new_cov_sqrt_)
        
    if G.comp_opt=="cholesky":
        #new_cov sqrt may already be calculated, see above
        if new_cov_sqrt_ is None:
            new_cov_sqrt_ = matrix_sqrt(new_cov_)
        new_logdet_ = 2*tf.reduce_sum(tf.log(tf.linalg.diag_part(new_cov_sqrt_)))
    
    elif G.comp_opt=="svd":
        s_new_cov, u_new_cov, v_new_cov = tf.linalg.svd(new_cov_)
        #new_cov sqrt may already be calculated, see above
        if new_cov_sqrt_ is None:
            s_new_cov_sqrt = tf.linalg.diag(tf.sqrt(s_new_cov))
            new_cov_sqrt_ = tf.matmul(u_new_cov, tf.matmul(s_new_cov_sqrt, v_new_cov, adjoint_b=True))
        new_logdet_ = tf.reduce_sum(tf.log(s_new_cov))
        
    return new_cov_, new_cov_sqrt_, new_logdet_

def sym(M):
    """
    Apply the "Sym" operation to a square matrix, the purpose being to make it symmetric.
    M ↦ M + M^T - diag(M)
    """
    return 0.5 * (M + tf.transpose(M)) - tf.diag(tf.linalg.diag_part(M))
def Sym(Ms):
    """
    Apply the "Sym" operation to a set of square matrices.
    """
    return tf.map_fn(sym, Ms,  dtype=tf.float32)
