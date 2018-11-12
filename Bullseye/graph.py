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
from .utils import *

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
        if not G.tf_dataset:
            X = tf.placeholder(tf.float32, name='X', shape = [None, d])
            Y = tf.placeholder(tf.float32, name='Y', shape = [None, k])
        else:
            filenames = [G.file]
            record_defaults = [tf.float32] * (d+1)
            dataset = tf.data.experimental.CsvDataset(filenames, record_defaults)
            batched_dataset = dataset.batch(G.chunksize)
            iterator = batched_dataset.make_initializable_iterator()

            it_next = iterator.get_next()
            X = tf.transpose(tf.convert_to_tensor(it_next[1:(d+1)]))
            Y = tf.one_hot(tf.cast(tf.transpose(it_next[0]),'int32'), k)

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


    #status
    status = tf.get_variable("status",[], initializer = tf.zeros_initializer, dtype = tf.string)

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
    new_cov_ = tf.linalg.inv(step_size * beta + (1-step_size) * tf.linalg.inv(cov))
    new_mu_  = mu - step_size * tf.einsum('ij,j->i', new_cov_, rho)

    new_cov = tf.get_variable("new_cov",[p,p],
                              initializer = tf.zeros_initializer,
                              dtype = tf.float32)
    new_mu = tf.get_variable("new_mu",[p],
                              initializer = tf.zeros_initializer,
                              dtype = tf.float32)


    update_new_cov = tf.assign(new_cov, new_cov_)
    update_new_mu = tf.assign(new_mu, new_mu_)

    #SVD decomposition of new_cov
    new_cov_S_, new_cov_U, new_cov_V = tf.linalg.svd(new_cov)
    new_cov_S_sqrt = tf.linalg.diag(tf.sqrt(new_cov_S_))
    new_cov_sqrt_ = tf.matmul(new_cov_U,
                             tf.matmul(new_cov_S_sqrt,new_cov_V, adjoint_b=True)) #[p,p]
    new_cov_S = tf.get_variable("new_cov_S", [p],
                                initializer = tf.zeros_initializer,
                                dtype=tf.float32)
    new_cov_sqrt = tf.get_variable("new_cov_sqrt", [p,p],
                                   initializer = tf.zeros_initializer,
                                   dtype = tf.float32)

    update_new_cov_S = tf.assign(new_cov_S, new_cov_S_)
    update_new_cov_sqrt = tf.assign(new_cov_sqrt, new_cov_sqrt_)

    if G.local_std_trick:
        update_new_parameters = [update_new_cov, update_new_mu, update_new_cov_S, update_new_cov_sqrt]
    else:
        update_new_parameters = [update_new_cov, update_new_mu]
    #sampling related
    z, z_weights = generate_sampling_tf(G.s, dim_samp)
    tf.identity(z, name = "z")
    tf.identity(z_weights, name = "z_weights")

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
    tf.identity(z_prior, name = "z_prior")
    tf.identity(z_weights_prior, name = "z_weights_prior")

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

    tf.identity(computed_e, name = "computed_e")
    tf.identity(computed_rho, name = "computed_rho")
    tf.identity(computed_beta, name = "computed_beta")


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

            #chunk as sum, will increase step by step global_e, global_rho and global_beta
            # with update_globals
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

            width = len(str(G.num_of_chunks))
            for _ in range(G.num_of_chunks):
                idx = "chunk_{_:0>{width}}".format(_=_, width=width)
                global_e.append(tf.get_variable("global_e_"+idx,
                                                [],
                                                initializer = tf.zeros_initializer,
                                                dtype = tf.float32))
                global_rho.append(tf.get_variable("global_rho_"+idx,
                                        [p],
                                        initializer = tf.zeros_initializer,
                                        dtype = tf.float32))
                global_beta.append(tf.get_variable("global_beta_"+idx,
                                        [p,p],
                                        initializer = tic(np.linalg.inv(G.cov_0)),
                                        dtype = tf.float32))

                update_global_e.append(tf.assign(global_e[_], computed_e + computed_e_prior))
                update_global_rho.append(tf.assign(global_rho[_], computed_rho + computed_rho_prior))
                update_global_beta.append(tf.assign(global_beta[_], computed_beta + computed_beta_prior))

        #chunk
        tf.identity(update_global_e, name = "update_global_e")
        tf.identity(update_global_rho, name = "update_global_rho")
        tf.identity(update_global_beta, name = "update_global_beta")

        global_list = []
        if G.tf_dataset:
            global_list += [iterator]
        if not G.chunk_as_sum:
            global_list += [global_e, global_rho, global_beta]

        init_globals = tf.variables_initializer(global_list)

        update_globals = [update_global_e, update_global_rho, update_global_beta]
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
    if not G.local_std_trick:
        logdet_cov = tf.linalg.logdet(new_cov)
    else :
        logdet_cov = tf.reduce_sum(tf.log(new_cov_S), axis = 0)

    new_ELBO = - new_e + 0.5 *  logdet_cov + 0.5 * np.log(2*np.pi*np.e)

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
        update_step_size = tf.assign(step_size, G.speed, name = "update_step_size")
        status_to_accepted = tf.assign(status, "accepted")

        with tf.control_dependencies([update_e, update_rho, update_beta,
                                     update_cov, update_mu, update_ELBO,
                                     update_step_size]):
            return [tf.assign(status, "accepted"), new_ELBO, ELBO]

    """
    REFUSED UPDATE
    """
    def refused_update():
        decrease_step_size = tf.assign(step_size, step_size*G.step_size_decrease_coef)
        status_to_refused = tf.assign(status, "refused")
        with tf.control_dependencies([decrease_step_size]):
            return [tf.assign(status, "refused"), new_ELBO, ELBO]

    """
    ITERATIONS
    """
    brutal_iteration = accepted_update
    soft_iteration = tf.cond(new_ELBO > ELBO, accepted_update, refused_update)

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
                'update_globals' : update_globals,
                'init_globals' : init_globals,
                'global_e' : global_e,

                'computed_e' : computed_e,
                'computed_e_prior' : computed_e_prior,
                'computed_rho' : computed_rho,
                'computed_rho_prior' : computed_rho_prior,
                'computed_beta' : computed_beta,
                'computed_beta_prior' : computed_beta_prior,

                'iteration' : iteration,
                'status' : status,

                'update_new_parameters' : update_new_parameters
                }

    return graph, ops_dict
