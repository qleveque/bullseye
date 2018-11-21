"""
The ``bullseye_graph`` module
=============================

Contains the definition of the Bullseye.Graph class, allowing to create a
graph with specific options and to launch the Bullseye algorithm.
See the ``Graph`` class below for more details.
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import os
import re
import time
import math

from .graph import construct_bullseye_graph
from .predefined_functions import compute_ps,\
    predefined_Phis, predefined_Psis, predefined_Projs
from .profilers import RunSaver
from .utils import *
from .warning_handler import *
from .visual import *
from .graph_aux import *

#remove tensorflow warning message at the beginning
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class Graph:
    """
    The ``Graph`` class
    ===================
    This class aims to apply the bullseye method. Instantiate it, configure the
    different options with which you want to run the algorithm, build the graph
    and finally execute it.

    In order to build the graph properly, there are different steps to follow.
    You first need to instantiate the class without any parameters.
    Then, you need to call three methods that will configure the fundamental
    inputs of the algorithm, namely ::
        ``feed_with`` to specify the data on which you will work,
        ``set_model`` or ``set_predefined_model`` to specify your model, and
        ``init_with`` to specify the starting parameters.

    Once this is done, you can optionally select additional options using
    ``set_options`` to stipulate which variants of the algorithm you want to
    work with.

    You can then build and run the implicit ``tensorflow`` graph calling
    respectively the ``build`` and ``run`` methods.

    Attributes
    ----------
    #→

    Example
    -------
    >>> import Bullseye
    >>> bull = Bullseye.Graph()
    >>> bull.feed_with(file = "dataset.csv", chunksize = 50, k=10)
    >>> bull.set_predefined_model("multilogit")
    >>> bull.init_with(mu_0 = 0, cov_0 = 1)
    >>> bull.set_options(m = 0, compute_kernel = True)
    >>> bull.build()
    >>> bull.run()
    """

    def __init__(self):
        """
        Initialization of Bullseye_graph.
        Does not take any parameters.
        """

        #listing of all fundamental attributes for consistency
        attrs = [
            #the implicit tensorflow graph
            "graph","in_graph",
            #data related
            "d","k","p","X","Y","prior_std",
            "file","chunksize","nb_of_chunks", "number_of_chunk_max",
            #init related
            "mu_0","cov_0",
            #φ's, ψ's and projections related
            "Psi","grad_Psi","hess_Psi",
            "Phi","grad_Phi","hess_Phi","Proj",
            "use_projs"
            #saver related
            "saver"
            ]

        #listing of all option attributes and their default values
        #→ think about putting these options in the __init__ parameters
        options = {
                #if brutal iteration, the ELBO will be updated even if it
                #decreases
                "brutal_iteration"          : False,
                #default speed of the algorithm, γ will start with this value
                "speed"                     : 1,
                #γ will decrease as γ*step_size_decrease_coef when ELBO
                # did not increase enough
                "step_size_decrease_coef"   : 0.5,
                #number of sample to approximate expectations
                "s"                         : 10,
                #quadrature_deg is the degree of the gaussian hermite
                # quadrature used to approximated prior expectation
                "quadrature_deg"            : 12,
                #we have s observations of the activations. make flatten
                # activations to True in order to flatten the observations
                # into a large unique observation.
                "flatten_activations"       : False,
                #when computing the local variances, compute the square roots
                # one by one or use a trick to compute only the square root
                # of cov
                "local_std_trick"           : True ,
                #number of batches for the likelihood
                "m"                         : 0,
                #number of batches for the prior
                "m_prior"                   : 0,
                #when computing ABA^T, compute the kernels H=AA^T for fast
                #computations of ABA^T, but exponential space is required
                "compute_kernel"            : False,
                #same as compute_kernel, but for the prior
                "compute_prior_kernel"      : False,
                #/!\ IN CONSTRUCTION /!\
                #tests to improve capacities with sparse matrices
                "sparse"                    : True,
                #when streaming a file, if chunk_as_sum is true, does not
                # keep track of the different values of eᵢ,ρᵢ,βᵢ in order
                # to save space
                "chunk_as_sum"              : True,
                #when prior covariance is diagonal, prevents the use of
                # exponential space and improves speed
                "keep_1d_prior"             : True,
                #when streaming a file, consider only a given number of
                # chunks
                ""       : 0,
                #use the natural value of eᵢ,ρᵢ,βᵢ centered in 0
                "natural_param_likelihood"  : False,
                #same as natural_param_likelihood for the prior
                "natural_param_prior"       : True,
                #make the run silent or not
                "silent"                    : True,
                #when streaming through a file, use tensorflow dataset class
                "tf_dataset"                : False,
                #/!\ IN CONSTRUCTION /!\
                #use einsum or dot_product when multiplying big matrices
                "use_einsum"                : True,
                #include timeliner in saved informations
                "timeliner"                 : False,
                #include tf profiler in saved informations,
                "profiler"                  : False,
                #include results of each epochs in saved informations
                "keep_track"                : False,
                #backtracking degree
                "backtracking_degree"       : -1
                }

        #keep in mind the name of those options
        self.option_list = list(options)

        #add the different attributes in the Graph class
        for key in attrs:
            setattr(self, key, None)
        for key in self.option_list:
            setattr(self, key, options[key])

        #attributes that ensure the correct construction of the graph
        self.feed_with_is_called = False
        self.set_model_is_called = False
        self.init_with_is_called = False
        self.build_is_called = False

    def feed_with(self, X = None, Y = None, d = None, k = None,
        file = None, chunksize = None, number_of_chunk_max = None):
        """
        Feed the graph with data. There are multiple ways of doing so.

        Method 1: requires X and Y
            Feed with a design matrix X and a response matrix Y.
        Method 2: requires d and k
            In the case X and Y are not defined yet, you only need to specify
            there sizes with d and k. In this case, X and Y need to be specified
            when ``run`` is called.
        Method 3: requires file, chunksize and k
            To stream a file.

        Parameters
        ----------
        X : np.array [None,d]
            Design matrix.
        Y : np.array [None, k]
            Response matrix.
        d : int
            d
        k : int
            k
        prior_std : int, np.array[p] or np.array [p,p]
            Prior std matrix. Can also be a vector or a int if it is diagonal.
        file : string
            Path of the file to stream (.csv format).
        chunksize : int
            Chunksize
        number_of_chunk_max : int
            Number of chunk to consider per iterations.
        """

        #method 1
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

        #method 2
        elif file is None:
            assert d is not None
            assert k is not None
            assert type(d), type(k) == [int,int]

            self.d = d
            self.k = k

        #method 3
        else:
            assert os.path.isfile(file)
            if chunksize is not None:
                assert type(chunksize)==int
            assert k is not None
            assert type(k) == int

            #retrieve the parameters
            self.k = k
            self.file = file
            self.chunksize = chunksize

            #deduce d
            #→ not a perfect method to assert d
            reader = pd.read_table(file, sep=",", chunksize = 1)
            for chunk in reader:
                data = list(chunk.shape)
                break
            self.d = data[-1]-1

            #deduce n
            #→ not a perfect method to assert n
            n = sum(1 for line in open(file))

            #deduce number_of_chunk_max
            if number_of_chunk_max is not None:
                self.nb_of_chunks = number_of_chunk_max
            else:
                self.nb_of_chunks = math.ceil(n/chunksize)

        #remember this method is called to ensure consistency and prevent errors
        self.feed_with_is_called = True

    def set_predefined_model(self, model,
        phi_option=None, proj_option=None, psi_option=None,
        prior_std = 1,
        use_projections = True,
        **specific_parameters):
        """
        →specific_parameters
        Make use of the ``predefined_functions`` module.
        Specify to the graph a given model. There are multiple ways of doing so.
        →
        Method 1: use_projections is set to True or not specified
            Then the implicit computations will be using Psi, and you can
            specify ``psi_option``.
        Method 2: use_projection is set to False
            Then the implicit computations will be using projections, and you
            can specify ``phi_option`` and ``proj_option``.

        Parameters
        ----------
        model: str, optional
            The model, e.g. "multilogit", see ``predefined_functions`` module.
        phi_option : str, optional
            phi_option, e.g. "mapfn", see ``predefined_functions`` module.
        proj_option : str, optional
            proj_option, e.g. "mapfn", see ``predefined_functions`` module.
        psi_option : str, optional
            psi_option, e.g. "std", see ``predefined_functions`` module.
        use_projections : bool, optional
            Assert whether we should use projections to simplify θ locally or
            not.
        """
        #keep that in mind
        self.use_projs = use_projections

        #compute p
        p = compute_ps[model](self.d, self.k, **specific_parameters)
        #method 1
        if use_projections:
            #basic suffix specifying the model
            suffix_phi = model
            suffix_proj = model

            #second suffix specifying the model options
            if phi_option is not None:
                suffix_phi += "_"+phi_option
            if proj_option is not None:
                suffix_proj+= "_"+proj_option

            #get the Phis
            Phi_, grad_Phi_, hess_Phi_ = Phis[suffix_phi]
            #consider specific_parameters
            Phi = lambda A,Y : Phi_(A,Y,**specific_parameters)
            grad_Phi = lambda A,Y : grad_Phi_(A,Y,**specific_parameters)
            hess_Phi = lambda A,Y : hess_Phi_(A,Y,**specific_parameters)

            #get proj
            Proj = Projs[suffix_proj]

            #use other method
            self.set_model(Phi=Phi, grad_Phi=grad_Phi, hess_Phi=hess_Phi,
                Proj=Proj, p=p)

        #method 2
        else:
            #basic suffix specifying the model
            suffix_psi = model

            #second suffix specifying the model options
            if psi_option is not None:
                suffix_psi += "_"+psi_option

            #get the Phis
            Psi_, grad_Psi_, hess_Psi_ = \
                predefined_Psis[suffix_psi]

            #be sure that the Psi function is not None
            assert Psi_ is not None

            #initialize Psi functions
            grad_Psi = None
            hess_Psi = None

            #consider specific_parameters
            Psi = lambda X,Y,theta : Psi_(X,Y,theta,**specific_parameters)
            if grad_Psi_ is not None:
                grad_Psi = lambda X,Y,theta : \
                        grad_Psi_(X,Y,theta,**specific_parameters)
            if hess_Psi_ is not None:
                hess_Psi = lambda X,Y,theta : \
                        hess_Psi_(X,Y,theta,**specific_parameters)

            #use other method
            self.set_model(Psi=Psi, grad_Psi=grad_Psi, hess_Psi=hess_Psi, p=p)

    def set_model(self,
        Psi = None, grad_Psi = None, hess_Psi = None,
        Phi = None, grad_Phi = None, hess_Phi = None, Proj = None,
        prior_std = 1,
        p = None):
        """
        Specify to the graph a given model.
        There are multiple ways of doing so.

        Method 1: requires Psi and p
            Manually choose the function ψ. You can also specify ∇ψ and Hψ,
            if not, ∇ψ and Hψ will be automatically computed using tf methods.
            The given functions take as parameters an a design matrix X, a
            response matrix Y, and the parameter θ.
        Method 2: requires Phi, Proj and p
            Manually choose the function φ. You can also specify ∇φ and Hφ,
            if not, ∇φ and Hφ will be automatically computed using tf methods.
            The given functions take as parameters an activation matrix and Y.

        The operations used within these functions must be tensorflow
        operations.
        """
        #set p
        self.p = p

        #method 1
        if Psi is not None:
            self.use_projs = False
            #ψ
            self.Psi = Psi
            #∇ψ
            if grad_Psi is None:
                self.grad_Psi = (lambda x,y,t: auto_grad_Psi(self.Psi,x,y,t))
            else:
                self.grad_Psi = grad_Psi
            #Hψ
            if hess_Psi is None:
                self.hess_Psi = (lambda x,y,t: auto_hess_Psi(self.Psi,x,y,t))
            else:
                self.hess_Psi = grad_Psi
        #method 2
        else:
            self.use_projs = True
            assert Phi is not None
            assert Projs is not None
            #ϕ
            self.Phi = Phi
            #∇ϕ
            if grad_Phi is None:
                self.grad_Phi = (lambda a,y: auto_grad_Phi(self.Phi,a,y))
            else:
                self.grad_Phi = grad_Phi
            #Hϕ
            if hess_Phi is None:
                self.grad_Phi = (lambda a,y: auto_hess_Phi(self.Phi,a,y))
            else:
                self.hess_Phi = hess_Phi

            #A
            self.Proj = Proj

        #handle std_prior
        #depending on the form of the given std_prior, transform it into a p×p
        #matrix.
        if type(prior_std) == int:
            self.prior_std = prior_std * np.eye(self.p)
        elif len(list(prior_std.shape)) == 1:
            assert list(prior_std.shape)==[self.p]
            self.prior_std = np.diag(prior_std)
        else:
            assert list(prior_std.shape) == [self.p, self.p]
            self.prior_std = prior_std

        #remember this method is called, to prevent errors
        self.set_model_is_called = True


    def init_with(self, mu_0 = 0, cov_0 = 1):
        """
        Specify μ₀ and Σ₀ from which the Bullseye algorithm should start.

        Parameters
        ----------
        mu_0 : float, or np.array [p], optional
            μ₀
        cov_0 : float, np.array [p] or [p,p], optional
            Σ₀
        """
        #handle μ₀
        if type(mu_0) in [float, int]:
            self.mu_0 = mu_0 * np.ones(self.p)
        elif len(mu_0.shape) == 1:
            assert list(mu_0.shape) == [p]
            self.mu_0 = mu_0

        #handle Σ₀
        if type(cov_0) in [float, int]:
            self.cov_0 = cov_0 * np.eye(self.p)
        elif len(cov_0.shape) == 1:
            assert list(mu_0.shape) == [self.p]
            self.cov_0 = np.diag(cov_0)
        else:
            assert list(mu_0.shape) == [self.p,self.p]
            self.cov_0 = cov_0

        #remember this method is called, to prevent errors
        self.init_with_is_called = True

    def set_options(self, **kwargs):
        """
        You can optionally select additional options using this method to
        stipulate which variants of the algorithm you want to work with.

        Parameters
        ----------
        **kwargs :
            The different options to be specified.
            →
        """

        #test if all the given options are allowed
        for key in list(kwargs):
            if key not in self.option_list:
                warn_unknown_parameter(key,
                                      function = "Bullseye_graph.set_options()")
            else:
                setattr(self, key, kwargs[key])

        #inform the user when some of these options are not compatible
        #→
        if self.keep_1d_prior:
            if "compute_prior_kernel" in list(kwargs):
                warn_useless_parameter("computed_prior_kernel", "keep_1d_prior",
                                    function = "Bullseye_graph.set_options()")
            self.compute_prior_kernel = False

    def build(self):
        """
        Builds the implicit tensorflow graph. Make use of the
        ``construct_bullseye_graph`` module.
        Does not take any parameters.
        """

        #to prevent error, ensures ``feed_with``, ``set_model`` and
        #``init_with`` methods have been called
        assert self.feed_with_is_called and self.set_model_is_called \
            and self.init_with_is_called

        #construct the graph
        self.graph, self.in_graph = construct_bullseye_graph(self)

        #remember this method is called, to prevent errors
        self.build_is_called = True

    def run(self, n_iter = 10, run_id = None, X = None, Y = None, save = False):
        """
        Run the implicit tensorflow graph.

        Parameters
        ----------
        n_iter : int, optional
            The number of iterations of the Bullseye algorithm
        run_id : str, optional
            The name of the current run, will in particular determine the name
            of the result directory.
        X : np.array [None, d], optional
            Allows the end user to specify a X if it hasn't been yet
        Y : np.array [None, k], optional
            Allows the end user to specify a Y if it hasn't been yet
        save : bool, optional
            Assert whether the results will be saved in the working directory or
            not

        Returns
        ------
        dict
            μ, Σ, ELBO and statistics in a dictionnary
        """

        #to prevent errors, ensures the graph is already built
        assert self.build_is_called

        #easy access to the tensorflow graph operations
        ops = self.in_graph

        #handle run_id
        if run_id is None:
            run_id = time.strftime("%Y_%m_%d_%H_%M_%S", time.gmtime())
        else:
            if not re.match("^[\\w]+$", run_id):
                err_bad_name(run_id)

        #feeding dict that will be used
        d_computed = {}
        #other run arguments
        run_kwargs = {}

        #handle data in argument
        if self.file is None:
            #note that e,rho and beta will in this case be computed directly
            #without changing parameters
            if X is not None and Y is not None:
                d_computed['X:0'] = X
                d_computed['Y:0'] = Y
            else:
                assert self.X is not None
                assert self.Y is not None

        #for saving
        if save:
            self.saver = RunSaver(self, run_id, run_kwargs, self.timeliner,
                            self.profiler, self.keep_track)

        print_title('beginning run "{}"'.format(run_id))

        #start the session
        with tf.Session(graph = self.graph) as sess:
            #initialize the graph
            sess.run(ops["init"],**run_kwargs)

            #starting iterations
            for epoch in range(n_iter):
                print_subtitle("epoch number {}".format(epoch))
                #---->start epoch
                if self.saver is not None:
                    self.saver.start_epoch()

                #compute new_mu, new_cov and optionaly more
                self.__run(sess, ops["update_new_parameters"],
                                    **run_kwargs)

                if self.file is not None:
                    #read chunks, and update global e, rho and beta
                    # in consequence
                    self.set_globals_from_chunks(sess, run_kwargs=run_kwargs)

                #compute new elbo
                statu, elbo, best_elbo = \
                    self.__run(sess, ops["iteration"],
                                        feed_dict = d_computed, **run_kwargs)
                statu = statu.decode('utf-8')

                #---->finish epoch
                if self.saver is not None:
                    self.saver.finish_epoch(statu, elbo, best_elbo)
                    if self.saver.keep_track:
                        mu, cov = sess.run([ops["mu"], ops["cov"]])
                        self.saver.save_step(mu,cov,epoch)

                print('{statu}, with {elbo}'.format(statu = statu, elbo = elbo))

            #get the lasts mu, cov, elbo
            final_mu, final_cov, final_elbo = \
                sess.run([ops["mu"], ops["cov"], ops["ELBO"]])

            #save
            if self.saver is not None:
                self.saver.save_final_results(mu=final_mu, cov=final_cov)

        print_end("end of the run")

        return {'mu':final_mu, 'cov':final_cov,'elbo':final_elbo,
                **self.saver.final_stats()}

    def __run(self, sess, ops_to_compute, feed_dict={}, **run_kwargs):
        """
        This private method can be seen as an overload of the tensorflow.Session
        ``run`` method. It is overloaded to assert consistency of the saver in a
        bullseye run.

        Parameters
        ----------
        sess : tensorflow.Session
            The current tensorflow session.
        ops_to_compute : list of tensorflow operations
            The list of operations to compute.
        feed_dict : dict
            A dictionnary with which to feed the session.
        **run_kwargs :
            Additional arguments that will be added to ``sess.run()``

        Returns
        ------
        dict
            The dictionnary returned by ``sess.run()``
        """

        if self.saver is not None:
            self.saver.before_run()

        d = sess.run(ops_to_compute,feed_dict = feed_dict, **run_kwargs)

        if self.saver is not None:
            self.saver.after_run(run_kwargs)
        return d

    def set_globals_from_chunks(self, sess, run_kwargs):
        """
        Compute global_e, global_rho and global_beta while streaming through the
        file.

        Parameters
        ----------
        sess : tensorflow.Session
            The current tensorflow session.
        **run_kwargs :
            Additional arguments that will be added to ``sess.run()``
        """

        #for simplicity
        ops = self.in_graph

        #initialize global e, rho and beta
        self.__run(sess,ops["init_globals"],**run_kwargs)

        #create a pandas reader to stream the file
        if not self.tf_dataset:
            reader = pd.read_table(self.file,
                                   sep = ",",
                                   chunksize = self.chunksize)

            #start streaming
            for (i,chunk) in enumerate(reader):

                #if the number of chunk to consider has been reached, we break
                if not i<self.nb_of_chunks:
                    break

                #decode data
                data = np.asarray(chunk)

                #→ transformations directly inside the run... not that good
                X = data[:,1:]/253.
                Y = to_one_hot(data[:,0])

                #create the feeding dict
                d_ = {"X:0" : X, "Y:0" : Y}

                #update the global parameters
                self.run_globals_update(sess, i, run_kwargs=run_kwargs, dict=d_)

        #else, create a tf.dataset reader to stream the file
        else:
            for i in range(self.nb_of_chunks):
                self.run_globals_update(sess, i, run_kwargs=run_kwargs)

    def run_globals_update(self, sess, i, run_kwargs, dict={}):
        """
        Update global_e, global_rho and global_beta with a given chunk.

        Parameters
        ----------
        sess : tensorflow.Session
            The current tensorflow session.
        i : int
            Chunk number.
        run_kwargs :
            Additional arguments that will be added to ``sess.run()``
        dict : dict
            A dictionnary with which to feed the session.
        """
        #for simplicity
        ops = self.in_graph

        #add to e,ρ and β the current eᵢ,ρᵢ and βᵢ
        if self.chunk_as_sum:
            self.__run(sess,ops["update_globals"], feed_dict = dict,
                    **run_kwargs)

        #chunk as list : append current eᵢ, ρᵢ and βᵢ to [eᵢ],[ρᵢ],[βᵢ]
        else:
            self.__run(sess,ops["update_globals"][i], feed_dict = dict,
                    **run_kwargs)

        print("Chunk number {} done.".format(i))
