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
        if not G.tf_dataset:
            X = tf.placeholder(tf.float32, name='X', shape = [None, d])
            Y = tf.placeholder(tf.float32, name='Y', shape = [None, k])
        else:
            filenames = [G.file]
            record_defaults = [tf.float32] * (d+1)
            dataset = tf.data.experimental.CsvDataset(filenames,record_defaults)
            batched_dataset = dataset.batch(G.chunksize)
            iterator = batched_dataset.make_initializable_iterator()

            it_next = iterator.get_next()

            if not G.to_one_hot:
                #→ to change, not that nice
                X = tf.transpose(tf.convert_to_tensor(it_next[k:(d+k)]))
                Y = tf.transpose(tf.convert_to_tensor(it_next[:k]))
            else:
                read_Y = tf.transpose(it_next[0])
                X = tf.transpose(tf.convert_to_tensor(it_next[1:(d+1)]))
                Y = tf.one_hot(tf.cast(read_Y, 'int32'), k)
    else:
        X = tf.get_variable("X", G.X.shape,initializer = tic(G.X),
                            dtype = tf.float32)
        Y = tf.get_variable("Y",G.Y.shape,initializer = tic(G.Y),
                            dtype = tf.float32)

    #prior_std related
    """
    →
    if G.keep_1d_prior:
        prior_std = tf.get_variable("prior_std",
                        [p],
                        initializer = tic(np.diag(G.prior_std)),
                        dtype = tf.float32)
    else:
        prior_std = tf.get_variable("prior_std",
                        G.prior_std.shape,
                        initializer = tic(G.prior_std),
                        dtype = tf.float32)
    """

    #status
    status = tf.get_variable("status",[], initializer = tf.zeros_initializer,
                            dtype = tf.string)

    #μ, Σ and ELBO related
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
    if G.backtracking_degree==-1:
        # Σⁿ⁺¹ = (γ·(Σₘ)⁻¹ + (1-γ)·(Σⁿ)⁻¹)⁻¹
        new_cov_=tf.linalg.inv(gamma * cov_max_inv \
                               + (1-gamma) * tf.linalg.inv(cov))
        new_cov_sqrt_ = None
    elif G.backtracking_degree==1:
        # Σⁿ⁺¹ = γ·Σₘ + (1-γ)·Σⁿ
        new_cov_ = gamma*(cov_max) + (1-gamma) * cov
        new_cov_sqrt_ = None
        
    elif G.backtracking_degree==0.5:
        #Sⁿ⁺¹ = γ·Σₘ^(½) + (1-γ)·Sⁿ                with Sⁿ = (Σⁿ)^½
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
    
    new_mu_  = mu - step_size * tf.einsum('ij,j->i', beta_inv, rho)

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
    
    #sampling hermite for normal iid prior
    #→
    z_hermite = None
    weights_hermite = None

    """
    TRIPLETS
    """
    #LIKELIHOOD TRIPLET
    #for readability
    ltargs = [new_mu,new_cov,new_cov_sqrt]
    
    #if not batched
    if not G.m>0:
        computed_e, computed_rho, computed_beta = \
            likelihood_triplet(G,X,Y,*ltargs)
    #if batched
    else :
        computed_e, computed_rho, computed_beta = \
            batched_likelihood_triplet(G,X,Y,*ltargs)

    tf.identity(computed_e, name = "computed_e")
    tf.identity(computed_rho, name = "computed_rho")
    tf.identity(computed_beta, name = "computed_beta")


    #PRIO TRIPLET
    #for readability
    ptargs = [new_mu,new_cov,new_cov_sqrt,z_hermite,weights_hermite]
    #if not batched
    computed_e_prior, computed_rho_prior, computed_beta_prior =\
        prior_triplet(G, *ptargs)

    tf.identity(computed_e_prior, name = "computed_e_prior")
    tf.identity(computed_rho_prior, name = "computed_rho_prior")
    tf.identity(computed_beta_prior, name = "computed_beta_prior")

    """
    FOR CHUNKS
    """

    if G.file is not None:
        #if streaming through a file as sum
        if G.chunk_as_sum:
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

            #chunk as sum, will increase step by step global_e, global_rho and
            #global_beta with update_globals
            update_global_e = tf.assign(global_e,
                        global_e + computed_e + computed_e_prior)
            update_global_rho = tf.assign(global_rho,
                        global_rho + computed_rho + computed_rho_prior)
            update_global_beta = tf.assign(global_beta,
                        global_beta + computed_beta + computed_beta_prior)


        #chunk as list : the sum will be computed outside of the graph
        else:
            global_e = []
            global_rho = []
            global_beta = []

            update_global_e = []
            update_global_rho = []
            update_global_beta = []

            width = len(str(G.nb_of_chunks))
            for _ in range(G.nb_of_chunks):
                idx = "chunk_{_:0>{width}}".format(_=_, width=width)
                global_e.append(tf.get_variable("global_e_"+idx, [],
                                        initializer = tf.zeros_initializer,
                                        dtype = tf.float32))
                global_rho.append(tf.get_variable("global_rho_"+idx, [p],
                                        initializer = tf.zeros_initializer,
                                        dtype = tf.float32))
                global_beta.append(tf.get_variable("global_beta_"+idx, [p,p],
                                        initializer=tic(np.linalg.inv(G.cov_0)),
                                        dtype = tf.float32))

                update_global_e.append(tf.assign(global_e[_],
                                        computed_e + computed_e_prior))
                update_global_rho.append(tf.assign(global_rho[_],
                                        computed_rho + computed_rho_prior))
                update_global_beta.append(tf.assign(global_beta[_],
                                        computed_beta + computed_beta_prior))

        #chunk
        tf.identity(update_global_e, name = "update_global_e")
        tf.identity(update_global_rho, name = "update_global_rho")
        tf.identity(update_global_beta, name = "update_global_beta")

        global_list = []
        if G.tf_dataset:
            global_list += [iterator]
        if G.chunk_as_sum:
            global_list += [global_e, global_rho, global_beta]

        init_globals = tf.variables_initializer(global_list)

        update_globals=[update_global_e, update_global_rho, update_global_beta]
    #if not streaming through a file, they will not be used
    else:
        global_e, global_rho, global_beta = 3*[tf.no_op()]
        update_globals, init_globals = 2*[tf.no_op()]


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
        if G.chunk_as_sum:
            new_e = global_e
            new_rho = global_rho
            new_beta = global_beta
        else: #chunk as list
            new_e = tf.reduce_sum(global_e, axis = 0)
            new_rho = tf.reduce_sum(global_rho, axis = 0)
            new_beta = tf.reduce_sum(global_beta, axis = 0)

    tf.identity(new_e, name = "new_e")
    tf.identity(new_rho, name = "new_rho")
    tf.identity(new_beta, name = "new_beta")

    #new ELBO
    H = 0.5 *  new_logdet + d * 0.5 * np.log(2*np.pi*np.e)
    new_ELBO = - new_e + H

    tf.identity(new_ELBO, name = "new_ELBO")

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
            return [tf.assign(status, bcolors.OKGREEN+"accepted"+bcolors.ENDC),
                new_ELBO, ELBO]

    """
    REFUSED UPDATE
    """
    def refused_update():
        decrease_step_size = tf.assign(step_size,
                                step_size*G.step_size)
        #status_to_refused = tf.assign(status, "refused")
        with tf.control_dependencies([decrease_step_size]):
            return [tf.assign(status, bcolors.FAIL+"refused"+bcolors.ENDC),
                new_ELBO, ELBO]

    """
    ITERATIONS
    """

    if G.eigmin_condition:
        #the eigmin_condition requests that eigmin(βⁿ⁺¹) > ½·eigmin(βⁿ)
        #note that:
        #   eigmin(βⁿ⁺¹) = min(new_cov_S⁻¹) = max(new_cov_S)⁻¹
        #   eigmin(βⁿ) = min(beta_S)
        #new_eigmin = tf.linalg.inv(tf.reduce_max(new_cov_S))
        #curr_eigmin = tf.reduce_min(beta_S)
        #condition_update = (new_ELBO > ELBO) and (2*new_eigmin > curr_eigmin)
        pass
    else:
        condition_update = new_ELBO > ELBO


    brutal_iteration = accepted_update
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

                'update_globals' : update_globals,
                'init_globals' : init_globals,
                'global_e' : global_e,

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

                'update_new_parameters' : update_new_parameters
                }

    return graph, ops_dict
