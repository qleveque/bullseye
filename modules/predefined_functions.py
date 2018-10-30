"""
    The ``predefined_functions`` module
    ======================
 
    Contains all functions related to the computation of ϕ
    when computing the log-posterior ψ(θ) = ∑φᵢ(θ) = ∑ϕ(Aᵢθ).
 
    :Example:
 
    >>> import functions
    >>> phi_, grad_phi_, hess_phi_, A_ = get_predefined_functions("multilogit", A_opt = "mapfn")
 
    Existing variants
    -----------------

    "matrix", "map", "map_opt"
"""
import numpy as np
import tensorflow as tf
 
def Softmax_probabilities(A):
    """
    Compute the softmax probabilities of each a in A.
    
    :param A: Score matrix.
    :type A: tf.tensor [n,k]
    :return: Softmax probabilities.
    :rtype: tf.tensor [n,k]
    """
    #remove max for numerical stability
    sp = tf.exp(A - tf.expand_dims(tf.reduce_max(A, reduction_indices=[1]),1))
    sp = sp/tf.expand_dims(tf.reduce_sum(sp, reduction_indices=[1]),1)
    return sp
    
def softmax_probabilities(a):
    """
    Compute the softmax probabilities a.
    
    :param a: Score vector.
    :type a: tf.tensor [k]
    :return: Softmax probabilities.
    :rtype: tf.tensor [k]
    """
    k = a.get_shape().as_list()[0]
    #remove max for numerical stability
    sp = tf.exp(a - tf.ones(k) * tf.reduce_max(a, reduction_indices=[0]))
    sp = sp/tf.reduce_sum(sp, reduction_indices=[0]) #[n,k]
    return sp


"""
    Core functions

    Refers to phi_*(), grad_phi_*() and hess_phi_*().
    
    The definitions of respectively ϕ, ∇ϕ, Hϕ when expliciting
    the log posterior as ψ(θ) = ∑φᵢ(θ) = ∑ϕ(Aᵢθ).
    
    * = "_[model](_[method])?"
    model ∈ {multilogit}
    opt ∈ {opt, der}
    
    :param a: Activation vector
    :param y: Response
    (:param p: optimization parameter)
    :type a: tf.tensor [k]
    :type y: tf.tensor [k]
    (:type p: tf.tensor [k])
    :return: ϕ(a) – ∇ϕ(a) – Hϕ(a)
    :rtype: tf.tensor [] – [k] – [k,k]

    .. note:: The operations used within these functions must be tensorflow
                operations.
    .. warning:: Using these functions implies the use of map_fn which is
                not optimized.
    .. seealso:: Core functions (Matrix form)
"""

"""
MULTILOGIT
"""
#no options
def phi_multilogit(a,y):
    p = softmax_probabilities(a)
    return -tf.log(tf.tensordot(y,p,1))
def grad_phi_multilogit(a,y):
    p = softmax_probabilities(a)
    return -y + p
def hess_phi_multilogit(a,y):
    p = softmax_probabilities(a)
    return tf.diag(p) - tf.tensordot(p,p,0)

#opt
def phi_multilogit_opt(a,y,p):
    return -tf.log(tf.tensordot(y,p,1))
def grad_phi_multilogit_opt(a,y,p):
    return -y + p
def hess_phi_multilogit_opt(a,y,p):
    return tf.diag(p) - tf.tensordot(p,p,0)

#aut_diff
def phi_multilogit_aut_diff(a,y):
    p = softmax_probabilities(a)
    return -tf.log(tf.tensordot(y,p,1))
def grad_phi_multilogit_aut_diff(a,y):
    return tf.gradients(phi_multilogit(a,y), a)[0]
def hess_phi_multilogit_aut_diff(a,y):
    return tf.hessians(phi_multilogit(a,y), a)[0]

"""
    Core functions (Matrix form)

    Refers to Phi_*(), grad_Phi_*() and hess_Phi_*().
    
    Matrix form of the core functions phi_*(), grad_phi_*() ans hess_phi_*().
    Corresponds to (ƒ(a,y) : (a,y)∈{A,Y}), for ƒ=phi_*(), grad_phi_*(), hess_phi_*().
    
    * = "_[model](_[method])?"
    model ∈ {multilogit}
    method ∈ {mapfn, mapfn_opt}
    
    :param A: Activation matrix
    :param Y: Response matrix
    :type A: tf.tensor [n,k]
    :type Y: tf.tensor [n,k]
    :return: ϕ(A) – ∇ϕ(A) – Hϕ(A)
    :rtype: tf.tensor [n] – [n,k] – [n,k,k]
    
    .. note:: The operations used within these functions must be tensorflow
                operations.
    .. warning:: Using these functions implies the use of map_fn which is
                not optimized.
"""

"""
MULTILOGIT
"""
#no options
def Phi_multilogit(A,Y):
    P=Softmax_probabilities(A)
    return -tf.log(tf.einsum('nk,nk->n',Y,P))
def grad_Phi_multilogit(A,Y):
    return -Y + Softmax_probabilities(A)
def hess_Phi_multilogit(A,Y):
    k = Y.get_shape().as_list()[-1]
    P=Softmax_probabilities(A)
    return tf.einsum('nk,kj->nkj',P,tf.eye(k)) - tf.einsum('ni,nj->nij',P,P)

#aut_grad
def Phi_multilogit_aut_grad(A,Y):
    return -tf.log(tf.einsum('nk,nk->n',Y,Softmax_probabilities(A)))
def grad_Phi_multilogit_aut_grad(A,Y):
    return tf.gradients(Phi_multilogit(A,Y),A)[0]
def hess_Phi_multilogit_aut_grad(A,Y):
    P=Softmax_probabilities(A)
    return tf.einsum('nk,kj->nkj',P,tf.eye(k)) - tf.einsum('ni,nj->nij',P,P)
    
#mapfn_opt
def Phi_multilogit_mapfn_opt(A,Y):
    P=Softmax_probabilities(A)
    return tf.map_fn(lambda x: phi_multilogit_opt(x[0], x[1], x[2]), (A,Y,P), dtype=tf.float32)
def grad_Phi_multilogit_mapfn_opt(A,Y):
    P=Softmax_probabilities(A)
    return tf.map_fn(lambda x: grad_phi_multilogit_opt(x[0], x[1], x[2]), (A,Y,P), dtype=tf.float32)
def hess_Phi_multilogit_mapfn_opt(A,Y):
    P=Softmax_probabilities(A)
    return tf.map_fn(lambda x: hess_phi_multilogit_opt(x[0], x[1], x[2]), (A,Y,P), dtype=tf.float32)

#mapfn
def Phi_multilogit_mapfn(A,Y):
    return tf.map_fn(lambda x: phi_multilogit(x[0], x[1]), (A,Y), dtype=tf.float32)
def grad_Phi_multilogit_mapfn(A,Y):
    return tf.map_fn(lambda x: grad_phi_multilogit(x[0], x[1]), (A,Y), dtype=tf.float32)
def hess_Phi_multilogit_mapfn(A,Y):
    return tf.map_fn(lambda x: hess_phi_multilogit(x[0], x[1]), (A,Y), dtype=tf.float32)

#mapfn_aut_diff
def Phi_multilogit_mapfn_aut_diff(A,Y):
    return tf.map_fn(lambda x: phi_multilogit_aut_diff(x[0], x[1]), (A,Y), dtype=tf.float32)
def grad_Phi_multilogit_mapfn_aut_diff(A,Y):
    return tf.map_fn(lambda x: grad_phi_multilogit_aut_diff(x[0], x[1]), (A,Y), dtype=tf.float32)
def hess_Phi_multilogit_mapfn_aut_diff(A,Y):
    return tf.map_fn(lambda x: hess_phi_multilogit_aut_diff(x[0], x[1]), (A,Y), dtype=tf.float32)
    
"""
    Projection
    
    Refers to proj_*()
    
    When expliciting, the log posterior as ψ(θ) = ∑φᵢ(θ) = ∑ϕ(Aᵢθ), corresponds to Aᵢ for observation xᵢ.

    :param x: One observation of the data
    :type x: tf.tensor [d]
    :return: Aᵢ corresponding to input xᵢ
    :rtype: tf.tensor [d, k]

    .. note:: The operations used within these functions must be tensorflow
                operations.
    .. warning:: Using these functions implies the use of map_fn which is
                not optimized.
    .. seealso:: Projection (matrix form)
"""
def proj_multilogit(x, d, k):
    #DEPRECATED
    #return tf.transpose(tf.convert_to_tensor((np.eye(k),tf.reshape(x,(1,d)))))
    pass

"""
    Projection (matrix form)
    
    Refers to Proj_*
    
    The matrix form of the projections proj_*()
    Corresponds to (proj_*(xᵢ) : i∈〚1,n〛).

    :param X: The design matrix.
    :type X: tf.tensor [n,d]
    :return: [Aᵢ : i∈〚1,n〛]
    :rtype: tf.tensor [n, d, k]

    .. note:: The operations used within these functions must be tensorflow
                operations.
"""

def Proj_multilogit(X,d,k):
    #KP = tf.linalg.LinearOperatorKronecker(eye, ones)
    X_tiled = tf.tile(X, [1,k])
    KP = tf.convert_to_tensor(np.kron(np.eye(k),np.ones((d,1))), tf.float32)
    return tf.einsum('np,pk->npk',X_tiled,KP)

def Proj_multilogit_mapfn(X, d, k):
    return tf.map_fn(lambda x: proj_multilogit(x,d,k), X)

"""
    FUNCTIONS TO CALL
"""

def get_predefined_functions(model, phi_option="", proj_option="", **kwargs):
    """
        TODO
    """
    m = globals().copy()
    m.update(locals())
    
    s_Phi = "Phi_"+model
    s_grad_Phi = "grad_Phi_"+model
    s_hess_Phi = "hess_Phi_"+model
    s_A = "Proj_"+model
    
    if phi_option!="":
        s_Phi += '_'+phi_option
        s_grad_Phi += '_'+phi_option
        s_hess_Phi += '_'+phi_option
    if proj_option!="":
        s_A += '_'+proj_option
    
    f_names = [s_Phi, s_grad_Phi, s_hess_Phi, s_A]
    r_Phi, r_grad_Phi, r_hess_Phi, r_A = [m.get(f_name) for f_name in f_names]
    
    f = [r_Phi, r_grad_Phi, r_hess_Phi, r_A]
    
    for i in range(len(f)):
        if not f[i]:
             raise NotImplementedError("Method %s not implemented" % f_names[i])
    return f
