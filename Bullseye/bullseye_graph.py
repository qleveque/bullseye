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
    predefined_Phis, predefined_Psis,\
    predefined_Projs, predefined_Pis,\
    predefined_Predicts
from tensorflow.initializers import constant as tic
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
    >>> bull.set_options()
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
            "d","k","p","X","Y",
            "file","m","M","to_one_hot",
            #init related
            "mu_0","cov_0",
            #φ's, ψ's and projections related
            "Psi","grad_Psi","hess_Psi",
            "Phi","grad_Phi","hess_Phi","Proj",
            "use_projs",
            #π's related
            "Pi","grad_Pi","hess_Pi",
            "prior_iid",
            #saver related
            "saver"
            ]

        #listing of all option attributes and their default values
        #→ think about putting these options in the __init__ parameters
        options = {
                #if brutal iteration, the ELBO will be updated even if it
                # decreases
                "brutal_iteration"          : False,
                #default speed of the algorithm, γ will start with this value
                "speed"                     : 1,
                #γ will decrease as γ*step_size_decrease_coef when ELBO
                # did not increase enough
                "step_size"   : 0.5,
                #number of sample to approximate expectations
                "s"                         : 50,
                #we have s observations of the activations. make flatten
                # activations to True in order to flatten the observations
                # into a large unique observation.
                "flatten_activations"       : False,
                #when computing the local variances, compute the square roots
                # one by one
                "local_std_trick"           : True,
                #when streaming a file, if chunk_as_sum is true, does not
                # keep track of the different values of eᵢ,ρᵢ,βᵢ in order
                # to save space
                "chunk_as_sum"              : True,
                #when streaming through a file, use tensorflow dataset class
                "tf_dataset"                : False,
                #include timeliner in saved informations
                "timeliner"                 : False,
                #include tf profiler in saved informations,
                "profiler"                  : False,
                #include results of each epochs in saved informations
                "keep_track"                : True,
                #backtracking degree
                "backtracking_degree"       : 1,
                #comp_opt
                "comp_opt"                  : "cholesky",
                #compute_gamma : equation to ensure positive semi-definite
                "compute_gamma"             : False,
                #autograd
                "compute_grad"              : "tf",
                #autohess
                "compute_hess"              : "tf",
                #diag_cov
                "diag_cov"                  : False
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
        self.set_prior_is_called = False
        self.init_with_is_called = False
        self.build_is_called = False

    def feed_with(self, X = None, Y = None, d = None, k = None,
        file = None, m = None, M = None, to_one_hot = False):
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

        →

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
        file : string
            Path of the file to stream (.csv format).
        chunksize : int
            Chunksize
        number_of_chunk_max : int
            Number of chunk to consider per iterations.
        """
        
        self.to_one_hot = to_one_hot
        
        #method 1
        if X is not None or Y is not None:
            assert X is not None
            assert Y is not None
            assert Y.shape[0] == X.shape[0]
            assert len(X.shape) in [1,2]
            assert len(Y.shape) in [1,2]

            if len(X.shape)==1:
                X = np.expand_dims(X,1)
            if len(Y.shape)==1:
                Y = np.expand_dims(Y,1)

            self.X = X
            self.Y = Y
            self.d = X.shape[-1]
            self.k = Y.shape[-1]
            self.m = None

        #method 2
        elif file is None:
            assert d is not None
            assert k is not None
            assert type(d), type(k) == [int,int]

            self.d = d
            self.k = k
            self.m = None

        #method 3
        else:
            assert os.path.isfile(file)
            assert k is not None
            assert type(k) == int

            #retrieve the parameters
            self.k = k
            self.file = file

            #deduce d
            #→ not a perfect method to assert d
            reader = pd.read_table(file, sep=",", chunksize = 1)
            for chunk in reader:
                data = list(chunk.shape)
                break
            if to_one_hot:
                self.d = data[-1]-1
            else:
                self.d = data[-1]-k

            #deduce n
            #→ not a perfect method to assert n
            n = sum(1 for line in open(file))

            if m is not None:
                assert type(m)==int
                self.m = m
            else:
                self.m = n

            #deduce number_of_chunk_max
            if M is not None:
                self.M = M
            else:
                self.M = math.ceil(n/self.m)

        #remember this method is called to ensure consistency and prevent errors
        self.feed_with_is_called = True

    def set_predefined_model(self, model,
        phi_option=None, proj_option=None, psi_option=None,
        use_projections = False,
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
        assert self.feed_with_is_called
        
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
            Phi_, grad_Phi_, hess_Phi_ = predefined_Phis[suffix_phi]
            
            #be sure that the Psi function is not None
            assert Phi_ is not None

            #initialize Psi functions
            grad_Phi = None
            hess_Phi = None
            
            #consider specific_parameters
            Phi = lambda A,Y : Phi_(A,Y,**specific_parameters)
            if grad_Phi_ is not None:
                grad_Phi = lambda A,Y : grad_Phi_(A,Y,**specific_parameters)
            if hess_Phi_ is not None:
                hess_Phi = lambda A,Y : hess_Phi_(A,Y,**specific_parameters)

            #get proj
            Proj = predefined_Projs[suffix_proj]
            assert Proj is not None
            
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

            #get the Psis
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
        assert self.feed_with_is_called
        
        #set p
        self.p = p

        #method 1
        if Psi is not None:
            self.use_projs = False
            assert Psi is not None
            #ψ
            self.Psi = Psi
            #∇ψ
            self.grad_Psi = grad_Psi
            if grad_Psi is None and self.compute_grad=="tf":
                self.grad_Psi = lambda X,Y,theta : \
                    tf.gradients(self.Psi(X,Y,theta),theta)[0]
            #Hψ
            self.hess_Psi = hess_Psi
            if hess_Psi is None and self.compute_hess=="tf":
                #self.hess_Psi = lambda A:tf.gradients(grad_Psi(A[0],A[1],A[2]),A[2])[0]
                self.hess_Psi = lambda X,Y,theta : \
                    tf.hessians(self.Psi(X,Y,theta),theta)[0]
        #method 2
        else:
            self.use_projs = True
            assert Phi is not None
            assert Proj is not None
            #ϕ
            self.Phi = Phi
            #∇ϕ
            self.grad_Phi = grad_Phi
            if grad_Phi is None and self.compute_grad=="tf":
                self.grad_Phi = lambda A, Y:tf.gradients(self.Phi(A,Y),A)[0]
            #Hϕ
            #impossible to use tf hessians
            self.hess_Phi = hess_Phi

            #A
            self.Proj = Proj

        #handle std_prior
        #depending on the form of the given std_prior, transform it into a p×p
        #matrix.
        """
        if type(prior_std) == int:
            self.prior_std = prior_std * np.eye(self.p)
        elif len(list(prior_std.shape)) == 1:
            assert list(prior_std.shape)==[self.p]
            self.prior_std = np.diag(prior_std)
        else:
            assert list(prior_std.shape) == [self.p, self.p]
            self.prior_std = prior_std
        """
        
        #remember this method is called, to prevent errors
        self.set_model_is_called = True

    def set_predefined_prior(self, prior, **specific_parameters):
        """
        →
        """
        #get the π's
        Pi_, grad_Pi_, hess_Pi_, iid = \
            predefined_Pis[prior]

        #be sure that the Pi function is not None
        assert Pi_ is not None

        #initialize others Pi functions
        grad_Pi = None
        hess_Pi = None
        #consider specific_parameters
        Pi = lambda theta : Pi_(theta,**specific_parameters)
        if grad_Pi_ is not None:
            grad_Pi = lambda theta : \
                    grad_Pi_(theta,**specific_parameters)
        if hess_Pi_ is not None:
            hess_Pi = lambda theta : \
                    hess_Pi_(theta,**specific_parameters)
        
        #use other method
        self.set_prior(Pi=Pi, grad_Pi=grad_Pi, hess_Pi=hess_Pi, iid = iid)

    def set_prior(self,Pi, grad_Pi = None, hess_Pi = None, iid = False):
        """
        →
        """
        #π
        self.Pi = Pi
        #∇π
        self.grad_Pi = grad_Pi
        if grad_Pi is None and self.compute_grad=="tf":
            self.grad_Pi = lambda theta:tf.gradients(Pi(theta),theta)[0]
        #Hπ
        self.hess_Pi = hess_Pi
        if hess_Pi is None and self.compute_grad=="tf":
            self.hess_Pi = lambda theta:tf.hessians(Pi(theta),theta)[0]
    
        #iid parameter
        self.prior_iid = iid
        
        #remember this method is called, to prevent errors
        self.set_prior_is_called = True


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
            assert list(mu_0.shape) == [self.p]
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
            
            if self.backtracking_degree==0.5:
                assert self.local_std_trick
            if self.compute_gamma:
                assert self.local_std_trick

        #inform the user when some of these options are not compatible
        #→
        if self.diag_cov:
            if self.prior_iid == False:
                #say to the user
                self.prior_iid = True

    def build(self):
        """
        Builds the implicit tensorflow graph. Make use of the
        ``construct_bullseye_graph`` module.
        Does not take any parameters.
        """

        #to prevent error, ensures ``feed_with``, ``set_model`` and
        #``init_with`` methods have been called
        assert self.feed_with_is_called and self.set_model_is_called \
            and self.init_with_is_called and self.set_prior_is_called

        #construct the graph
        self.graph, self.in_graph = construct_bullseye_graph(self)

        #remember this method is called, to prevent errors
        self.build_is_called = True

    def run(self, n_iter = 10, run_id = None, X = None, Y = None,
        debug_array = None):
        """
        →debug
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
            if not re.match("^[\\w ]+$", run_id):
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
                self.saver.start_epoch()
                
                #update new_mu, new_cov and optionaly more
                self.__run(sess, ops["update_new_parameters"],
                                    **run_kwargs)

                if self.file is not None:
                    #read chunks, and update partial es, rhos and betas
                    # in consequence
                    self.__set_partials_from_chunks(sess, run_kwargs=run_kwargs)
                #debug array-----
                if debug_array is not None:
                    ans = self.__run(sess,[ops[op] for op in debug_array],
                                     **run_kwargs)
                    for idx,an in enumerate(ans):
                        print(debug_array[idx])
                        print(an)
                #----------------
                #compute new elbo
                statu, elbo, best_elbo = \
                    self.__run(sess, ops["iteration"],
                                        feed_dict = d_computed, **run_kwargs)
                statu = statu.decode('utf-8')
                #---->finish epoch
                self.saver.finish_epoch(statu, elbo, best_elbo)
                if self.saver.keep_track:
                    mu, cov = sess.run([ops["mu"], ops["cov"]])
                    self.saver.save_step(mu,cov,epoch)

                b = bcolors.OKGREEN if statu=="accepted" else bcolors.FAIL
                print('{b}{statu}{e}, with {elbo}'.format(statu = statu, elbo = elbo,
                                                          e = bcolors.ENDC,
                                                          b = b))

            #get the lasts mu, cov, elbo
            final_mu, final_cov, final_elbo = \
                sess.run([ops["mu"], ops["cov"], ops["ELBO"]])

            #save
            self.saver.save_final_results(mu=final_mu, cov=final_cov)

        print_end("end of the run")

        #set the return dictionnary
        r =  {'mu':final_mu, 'cov':final_cov,'elbo':final_elbo}
        #add to the dictionnary what is saved by the saver
        r.update(self.saver.final_stats())
        return r

    def predict(self, X_test, mu, k, model = None, Predict = None,
                  **specific_parameters):
        """
        →
        """
        
        if Predict is None:
            assert model is not None
            Predict_ = predefined_Predicts[model]
            Predict = lambda X,mu : Predict_(X,mu,k,**specific_parameters)
        
        graph_test = tf.Graph()
        with graph_test.as_default() as g:
            mu = tf.get_variable("mu",mu.shape,
                    initializer = tic(mu),
                    dtype = tf.float32)
            X_test = tf.get_variable("X_test",X_test.shape,
                    initializer = tic(X_test),
                    dtype = tf.float32)
            init = tf.global_variables_initializer()
            
        with tf.Session(graph=graph_test) as sess:
            sess.run(init)
            T = sess.run(Predict(X_test,mu))
        return T
            

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

    def __set_partials_from_chunks(self, sess, run_kwargs):
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
        self.__run(sess,ops["init_chunks"],**run_kwargs)
        
        #create a pandas reader to stream the file
        if not self.tf_dataset:
            reader = pd.read_table(self.file,
                                   sep = ",",
                                   chunksize = self.m)

            #start streaming
            for (i,chunk) in enumerate(reader):

                #if the number of chunk to consider has been reached, we break
                if not i<self.M:
                    break
                
                #decode data
                data = np.asarray(chunk)
                #handle X and Y
                if self.to_one_hot:
                    X = data[:,1:]
                    Y = to_one_hot(data[:,0],self.k)
                else:
                    X = data[:,self.k:self.d+self.k]
                    Y = data[:,:self.k]
                
                #create the feeding dict
                d_ = {"X:0" : X, "Y:0" : Y}
                #update the global parameters
                self.__run_partials_update(sess, i, run_kwargs=run_kwargs, dict=d_)

        #else, create a tf.dataset reader to stream the file
        else:
            for i in range(self.M):
                self.__run_partials_update(sess, i, run_kwargs=run_kwargs)

    def __run_partials_update(self, sess, i, run_kwargs, dict={}):
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
            self.__run(sess,ops["update_partials"], feed_dict = dict,
                    **run_kwargs)

        #chunk as list : append current eᵢ, ρᵢ and βᵢ to [eᵢ],[ρᵢ],[βᵢ]
        else:
            self.__run(sess,ops["update_partials"][i], feed_dict = dict,
                    **run_kwargs)

        print("Chunk number {} done.".format(i))
