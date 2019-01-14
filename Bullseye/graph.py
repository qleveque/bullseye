"""
    The ``graph`` module
    ======================

    Contains the ``construct_bullseye_graph`` function which creates the
    implicit tensorflow graph.
"""
#→ may try to document this better

import tensorflow as tf
from tensorflow.initializers import constant as tic
import numpy as np
import re

from .sampling import *
from .graph_aux import *
from .utils import *

def construct_bullseye_graph(G):
    """
    Creates the implicit tensorflow graph given a Bullseye.Graph.
    This function is used in ``Graph.build``.

    Parameters
    ----------
    G : Bullseye.Graph
        Graph from which the tensorflow graph will be built.

    Returns
    -------
    graph : tf.graph
        The constructed tensorflow graph.
    ops_dict : dict
        A dictionnary containing the important variables and operations of the
        graph
    """
    #set the graph
    tf.reset_default_graph()
    graph = tf.get_default_graph()

    #for simplicity
    d,k,p = [G.d, G.k, G.p]
    dim_samp = p if not G.local_std_trick else k

    """
    VARIABLES
    """

    #X, Y related
    #always put Y as a [n,k] tensor
    if G.X is None and G.Y is None:
        if not G.tf_dataset: #placeholders of file with pandas
            X = tf.placeholder(tf.float32, name='X', shape = [None, d])
            Y = tf.placeholder(tf.float32, name='Y', shape = [None, k])
        elif G.file is not None: #file with tf.file
            filenames = [G.file]
            record_defaults = [tf.float32] * (d+k)
            dataset = tf.data.experimental.CsvDataset(filenames,record_defaults)
            batched_dataset = dataset.batch(G.m)
            iterator = batched_dataset.make_initializable_iterator()

            it_next = iterator.get_next()
            
            #not that nice...
            if not G.to_one_hot:
                X = tf.transpose(tf.convert_to_tensor(it_next[k:(d+k)]))
                Y = tf.transpose(tf.convert_to_tensor(it_next[:k]))
            else:
                read_Y = tf.transpose(it_next[0])
                X = tf.transpose(tf.convert_to_tensor(it_next[1:(d+1)]))
                Y = tf.one_hot(tf.cast(read_Y, 'int32'), k)
    else:
        if G.m is None:
            X = tf.get_variable("X", G.X.shape,initializer = tic(G.X),
                                dtype = tf.float32)
            Y = tf.get_variable("Y",G.Y.shape,initializer = tic(G.Y),
                                dtype = tf.float32)
                


    #status
    status = tf.get_variable("status",[], initializer = tf.zeros_initializer,
                            dtype = tf.string)

    #μ, Σ and ELBO related
    if G.diag_cov:
        G.cov_0 = np.einsum('ij,ij->ij',G.cov_0,np.eye(*list(G.cov_0.shape)))
    
    mu = tf.get_variable("mu",[p],initializer = tic(G.mu_0),
                        dtype = tf.float32)
    cov  = tf.get_variable("cov",[p,p],initializer = tic(G.cov_0),
                        dtype = tf.float32)

    ELBO = tf.get_variable("elbo",[],initializer = tic(-np.infty),
                        dtype = tf.float32)

    #e, ρ and β related
    e = tf.get_variable("e",[],initializer = tf.zeros_initializer,
                        dtype = tf.float32)
    rho = tf.get_variable("rho",[p], initializer = tf.zeros_initializer,
                          dtype = tf.float32)
    beta = tf.get_variable("beta", [p,p],
                           initializer = tic(np.linalg.inv(G.cov_0)),
                           dtype = tf.float32)

    #step size
    step_size = tf.get_variable("step_size", [], initializer = tic(G.speed),
                                dtype = tf.float32)

    #new_cov, new_mu
    new_cov = tf.get_variable("new_cov",[p,p],
                              initializer = tf.zeros_initializer,
                              dtype = tf.float32)
    new_mu = tf.get_variable("new_mu",[p],
                              initializer = tf.zeros_initializer,
                              dtype = tf.float32)
    new_logdet = tf.get_variable("newlogdet", [],
                              initializer = tf.zeros_initializer,
                              dtype = tf.float32)

    #new_cov_sqrt
    if (not G.local_std_trick or G.Psi is not None) or G.compute_gamma or not G.prior_iid:
        new_cov_sqrt = tf.get_variable("new_cov_sqrt", [p,p],
                                       initializer = tic(np.eye(p)),
                                       dtype = tf.float32)
    else:
        new_cov_sqrt = None
    
    if G.comp_opt is "cholesky":
        beta_chol = tf.linalg.cholesky(beta)
        beta_sqrt = tf.transpose(beta_chol)
        #beta_inv = tf.linalg.inv(beta)
        #beta_inv_sqrt = tf.linalg.inv(beta_sqrt)
        beta_inv = tf.transpose(tf.cholesky_solve(beta_chol, tf.eye(p)))
        beta_inv_sqrt = matrix_sqrt(beta_inv)
    elif G.comp_opt=="svd":
        s_beta, u_beta, v_beta = tf.linalg.svd(beta)
        s_beta_sqrt = tf.linalg.diag(tf.sqrt(s_beta))
        s_beta_inv = tf.linalg.diag(tf.reciprocal(s_beta))
        s_beta_inv_sqrt = tf.linalg.diag(tf.reciprocal(tf.sqrt(s_beta)))
        
        beta_sqrt = tf.matmul(u_beta, tf.matmul(s_beta_sqrt, v_beta, adjoint_b=True))
        beta_inv = tf.matmul(u_beta, tf.matmul(s_beta_inv, v_beta, adjoint_b=True))
        beta_inv_sqrt = tf.matmul(u_beta, tf.matmul(s_beta_inv_sqrt, v_beta, adjoint_b=True))

    #from definition of Σₘ
    cov_max = beta_inv
    cov_max_inv = beta
    cov_max_sqrt = beta_inv_sqrt

    assert G.backtracking_degree in [-1,1,0.5]

    if not G.compute_gamma:
        gamma = step_size
    else:
        #compute
        #K⁻¹=Σ^½•β•Σ^½
        K_inv = new_cov_sqrt @ beta @ new_cov_sqrt
        K_inv_sqrt = tf.linalg.cholesky(K_inv)
        K_eig = tf.math.reciprocal(tf.linalg.diag_part(K_inv))
        eig_limitation = 0.5
        K_eigmin = tf.reduce_min(K_eig)
        gamma = (eig_limitation - 1)/(K_eigmin -1)
    #backtracking
    pars = [cov_max_inv,new_cov_sqrt]
    new_cov_, new_cov_sqrt_, new_logdet_ = compute_new_cov_and_co(G,gamma,cov,cov_max,*pars)
    new_mu_  = mu - step_size * tf.einsum('ij,j->i', beta_inv, rho)

    """
    UPDATE
    """
    update_new_cov = tf.assign(new_cov, new_cov_)
    update_new_mu = tf.assign(new_mu, new_mu_)
    update_new_logdet = tf.assign(new_logdet, new_logdet_)

    update_new_parameters = [update_new_cov, update_new_mu, update_new_logdet]

    #new_cov_sqrt
    if new_cov_sqrt is not None:
        update_new_cov_sqrt = tf.assign(new_cov_sqrt, new_cov_sqrt_)
        update_new_parameters += [update_new_cov_sqrt]
    else:
        update_new_cov_sqrt = None

    """
    TRIPLETS
    """
    #LIKELIHOOD TRIPLET
    #for readability
    ltargs = [new_mu,new_cov,new_cov_sqrt]
    
    computed_e, computed_rho, computed_beta = \
        likelihood_triplet(G,X,Y,*ltargs)
    
    #test = test_(G,X,Y,*ltargs)

    #PRIO TRIPLET
    ptargs = [new_mu,new_cov,new_cov_sqrt]
    computed_e_prior, computed_rho_prior, computed_beta_prior =\
        prior_triplet(G, *ptargs)

    """
    FOR CHUNKS
    """
    if G.m is not None:
        #init partials
        # what should be updated each time we go again through the M chunks
        init_partials = []
        update_partials = []
        
        if G.file is not None and G.tf_dataset:
            #if we use tf.dataset, we need to reset the iterator to 0
            init_partials += [iterator]
        
        #consider sum
        if G.chunk_as_sum:
            e_sum = tf.get_variable("e_sum",[],
                                    initializer = tf.zeros_initializer,
                                    dtype = tf.float32)
            rho_sum = tf.get_variable("rho_sum",[p],
                                    initializer = tf.zeros_initializer,
                                    dtype = tf.float32)
            beta_sum = tf.get_variable("beta_sum",[p,p],
                                    initializer = tf.zeros_initializer,
                                    dtype = tf.float32)

            #chunk as sum, will increase step by step e_sum, rho_sum and
            #beta_sum with update_partials
            update_partial_e=tf.assign(e_sum,e_sum+computed_e)
            update_partial_rho=tf.assign(rho_sum,rho_sum+computed_rho)
            update_partial_beta=tf.assign(beta_sum,beta_sum+computed_beta)
            
            #we need to reset the sum to 0 at each iteration
            init_partials += [e_sum, rho_sum, beta_sum]

        #conside vectors
        # then we need to keep in mind all the different eᵢ,ρᵢ,βᵢ
        else:
            e_tab = []
            rho_tab = []
            beta_tab = []

            update_e_tab = []
            update_rho_tab = []
            update_beta_tab = []

            width = len(str(G.M))
            for _ in range(G.M):
                idx = "chunk_{_:0>{width}}".format(_=_, width=width)
                e_tab.append(tf.get_variable("e_"+idx, [],
                                        initializer = tf.zeros_initializer,
                                        dtype = tf.float32))
                rho_tab.append(tf.get_variable("rho_"+idx, [p],
                                        initializer = tf.zeros_initializer,
                                        dtype = tf.float32))
                beta_tab.append(tf.get_variable("beta_"+idx, [p,p],
                                        initializer=tf.zeros_initializer,
                                        dtype = tf.float32))

                update_partial_e.append(tf.assign(e_tab[_],computed_e))
                update_partial_rho.append(tf.assign(rho_tab[_],commuted_rho))
                update_partial_beta.append(tf.assign(beta_tab[_],computed_beta))

        update_partials+=[update_partial_e, update_partial_rho, update_partial_beta]
        init_chunks = tf.variables_initializer(init_partials)
    
    #if we do not consider batches
    else:
        init_chunks, update_partials, init_partials = 3*[tf.no_op()]


    """
    NEW PARAMS
    """
    #if G.file is None, we are directly using triplets results
    if G.m is None:
        new_e = computed_e + computed_e_prior
        new_rho  = computed_rho + computed_rho_prior
        new_beta = computed_beta + computed_beta_prior

    #if batches are used
    else:
        if G.chunk_as_sum:
            new_e = e_sum + computed_e_prior
            new_rho = rho_sum + computed_rho_prior
            new_beta = beta_sum + computed_beta_prior
        else: #chunk as list
            new_e = tf.reduce_sum(e_tab, axis = 0) + computed_e_prior
            new_rho = tf.reduce_sum(rho_tab, axis = 0) + computed_rho_prior
            new_beta = tf.reduce_sum(beta_tab, axis = 0) + computed_beta_prior

    #new ELBO
    H = 0.5 *  new_logdet + d * 0.5 * np.log(2*np.pi*np.e)
    new_ELBO = - new_e + H

    """
    ACCEPTED UPDATE
    """
    def accepted_update():
        update_e = tf.assign(e, new_e, name = "update_e")
        update_rho  = tf.assign(rho,  new_rho, name = "update_rho")
        update_beta = tf.assign(beta, new_beta, name = "update_beta")
        update_cov  = tf.assign(cov, new_cov, name = "update_cov")
        update_mu = tf.assign(mu, new_mu, name = "update_mu")
        update_ELBO = tf.assign(ELBO, new_ELBO, name = "update_ELBO")
        update_step_size=tf.assign(step_size, G.speed, name="update_step_size")

        #→
        with tf.control_dependencies([update_e, update_rho, update_beta,
                                     update_cov, update_mu, update_ELBO,
                                     update_step_size]):
            return [tf.assign(status, "accepted"), new_ELBO, ELBO]

    """
    REFUSED UPDATE
    """
    def refused_update():
        decrease_step_size = tf.assign(step_size,
                                step_size*G.step_size)
        #status_to_refused = tf.assign(status, "refused")
        with tf.control_dependencies([decrease_step_size]):
            return [tf.assign(status, "refused"), new_ELBO, ELBO]

    """
    ITERATIONS
    """
    condition_update = new_ELBO > ELBO

    brutal_iteration = accepted_update()
    soft_iteration = tf.cond(condition_update, accepted_update, refused_update)

    if G.brutal_iteration:
        iteration = brutal_iteration
    else:
        iteration = soft_iteration

    """
    INIT and RETURN
    """
    init = tf.global_variables_initializer()
    ops_dict = {'init' : init,
                'new_ELBO' : new_ELBO,
                'ELBO' : ELBO,
                'mu' : mu,
                'cov' : cov,
                'rho' : rho,
                'beta' : beta,
                'X' : X,
                'Y' : Y,
                'gamma' : gamma,
                
                'new_logdet' : new_logdet,
                'new_e': new_e,
                'new_rho' : new_rho,
                'new_beta': new_beta,

                'update_partials' : update_partials,
                'init_chunks' : init_chunks,

                'computed_e' : computed_e,
                'computed_e_prior' : computed_e_prior,
                'computed_rho' : computed_rho,
                'computed_rho_prior' : computed_rho_prior,
                'computed_beta' : computed_beta,
                'computed_beta_prior' : computed_beta_prior,

                'new_cov' : new_cov,
                'new_mu': new_mu,
                'new_cov_sqrt' : new_cov_sqrt,
                'update_new_cov' : update_new_cov,
                'update_new_cov_sqrt' : update_new_cov_sqrt,

                'iteration' : iteration,
                'status' : status,

                'update_new_parameters' : update_new_parameters,
                'init_partials' : init_partials
                #'test' : test
                }

    return graph, ops_dict
