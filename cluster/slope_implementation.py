
# coding: utf-8

import numpy as np
from sklearn.datasets import make_sparse_spd_matrix
from scipy import random, linalg
from scipy.stats import multivariate_normal
from tools.gm_tools import gm_params_generator, gaussian_mixture_sample
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.base import BaseEstimator
from sklearn.mixture import GMM
from sklearn.utils import check_array
from tools.matrix_tools import check_zero_matrix
from cvxpy import *

import os, sys
algo_root = '..'
sys.path.insert(0, algo_root)



def simplex_proj(z):
    """
    Projection sur le probability simplex
    http://arxiv.org/pdf/1309.1541.pdf
    :return:
    """
    # for reshaping from matrix type
    y = np.array(z).reshape(len(z))
    D, = y.shape
    x = np.array(sorted(y, reverse=True))
    u = [x[j] + 1. / (j + 1) * (1 - sum([x[i] for i in range(j + 1)])) for j in range(D)]
    l = []
    for idx, val in enumerate(u):
        if val > 0 :
            l.append(idx)
    if l == []:
        l.append(0)
    rho = max(l)
    lambd = 1. / (rho + 1) * (1 - sum([x[i] for i in range(rho + 1)]))
    return np.array([max(yi + lambd, 0) for yi in y])


def gradient_different_lambdas(X, means, covars, pi, lambd, EPSILON=1e-8):
    """
    Evaluate the gradient of -sum_i^n( log( sum_j^K (pi_j * phi(mu_j, sigma_j)(X_i) ))) + sum_l^K (lambda_l*pi_l)
    """
    densities = np.array([multivariate_normal.pdf(X, means[i], covars[i]) for i in range(len(pi))]).T
    #We reshape for the division and add EPSILON to avoid zero division
    #we add the lambda penality (SLOPE like)
    return -(densities/(((densities*pi).sum(axis=1)).reshape(X.shape[0],1) + EPSILON)).sum(axis=0) + lambd


def pi_differentLambdas_estim_fista(X, means, covars, pi, L, lambd):
    """
    We use FISTA to accelerate the convergence of the algorithm
    we project the next step of the gradient descent on the probability simplex
    """
    t_previous = 1
    pi_previous = np.copy(pi)
    xi = np.copy(pi_previous)
    # the number of iterations is given on FISTA paper,
    # we took ||pi_hat-pi_star||**2 = len(pi)**2
    fista_iter = int(np.sqrt(2*len(pi)**2 * L) // 1)
    for _ in range(min(500, fista_iter)):
        pi_next = simplex_proj(xi - 1./(np.sqrt(X.shape[0])*L)*gradient_different_lambdas(X, means, covars, xi, lambd))
        t_next = (1. + np.sqrt(1 + 4 * t_previous**2)) / 2
        xi = pi_next + (t_previous - 1) / t_next * (pi_next - pi_previous)
        pi_previous = np.copy(pi_next)
    return pi_next

def algo_different_lambdas_penalities_1(X, max_clusters, n_iter, L, alpha=0.01):
    """
    we inject in the gradient the penality, and project the estimate in the
    probability simplex
    """
    lambd = lambda_list_BH(max_clusters, alpha)
    # initialization of the algorithm
    g = GMM(n_components=max_clusters, covariance_type= "full")
    g.fit(X)
    means_estim, covars_estim, pi_estim = g.means_, g.covars_, g.weights_
    N = len(X)
    K = len(pi_estim)
    print "Init EM pi: ",pi_estim
    for it in range(n_iter):
        # We estimate pi according to the penalities lambdas given
        pi_estim = pi_differentLambdas_estim_fista(X, means_estim, covars_estim, pi_estim, L, lambd)
        # we remove the clusters with probability = 0
        non_zero_elements = np.nonzero(pi_estim)[0]
        K = len(non_zero_elements)
        pi_estim = np.array([pi_estim[i] for i in non_zero_elements])
        means_estim = np.array([means_estim[i] for i in non_zero_elements])
        covars_estim = np.array([covars_estim[i] for i in non_zero_elements])
        lambd = np.array([lambd[i] for i in non_zero_elements])
        # we estimate the conditional probability P(z=j/X[i])
        tau = tau_estim(X, means_estim, covars_estim, pi_estim)
        # Means
        means_estim = np.array([(tau[:, k]*X.T).sum(axis=1)*1/(N*pi_estim[k]) for k in range(K)])
        # covars
        covars_temp = np.array(
                [covar_estim(X, means_estim[k], tau[:, k], pi_estim[k]) for k in range(K)])
        non_empty_covar_idx = check_zero_matrix(covars_temp)
        pi_estim = [pi_estim[j] for j in non_empty_covar_idx]
        means_estim = [means_estim[j] for j in non_empty_covar_idx]
        covars_estim = [covars_estim[j] for j in non_empty_covar_idx]
        lambd = [lambd[j] for j in non_empty_covar_idx]
        K = len(pi_estim)
        if it%10 == 0 :
            print "iteration ",it, "pi: ", pi_estim
    return pi_estim, means_estim, covar_estim, tau_estim

def tau_estim(X, centers, covars, pi):
    try:
        densities = np.array([multivariate_normal.pdf(X, centers[k], covars[k], allow_singular=True) for k in range(len(pi))]).T * pi
        return (densities.T/(densities.sum(axis=1))).T
    except np.linalg.LinAlgError as e:
        print "Error on density computation for tau", e

def covar_estim(X, mean, tau, pi):
    """
    emp covariance of EM
    :param mean: mean for one cluster
    :param pi: pi for this cluster
    :param N: lenth of X
    :param tau: vector of proba for each X[i] in the cluster, given by tau[:,k]
    :return: emp covariance matrix of this cluster
    """
    N = len(X)
    Z = np.sqrt(tau).reshape(N, 1) * (X - mean)
    return 1 / (pi * N) * Z.T.dot(Z)

def check_zero_matrix(mat_list):
    """
    Return the list of matrices ids which are non empty
    :param mat_list: List of matrices, usually covariance matrices
    :return: list of ids of non empty matrices
    """
    non_zero_list = []
    for i in range(len(mat_list)):
        if np.count_nonzero(mat_list[i]) is not 0:
            non_zero_list.append(i)
    return non_zero_list

"""
def main():
    pi, means, covars = gm_params_generator(3,3)
    X,_ = gaussian_mixture_sample(pi, means, covars, 1e5)
    pi_e = algo_different_lambdas_penalities_1(X,max_clusters=5,n_iter=500, L=1e5)
    return pi_e, pi
pi_e, pi = main()
print "real pi: ", pi
print "estimated pi: ", pi_e
"""


def lambda_list_BH(K, alpha=1):
    #Cf pierre bellec, Candes (Mimimax SLOPE), lambda_BH, we add a normalization
    return alpha*np.array([np.sqrt(2./K*np.log(1.*K/(j+1))) for j in range(K)])


def simple_gradient(X, means, covars, pi, EPSILON=1e-8):
    """
    Evaluate the gradient of -sum_i^n( log( sum_j^K (pi_j * phi(mu_j, sigma_j)(X_i) )))
    """
    densities = np.array([multivariate_normal.pdf(X, means[i], covars[i]) for i in range(len(pi))]).T
    #We reshape for the division and add EPSILON to avoid zero division
    #we add the lambda penality (SLOPE like)
    return -(densities/(((densities*pi).sum(axis=1)).reshape(X.shape[0],1) + EPSILON)).sum(axis=0)

def ordered_optim_proj(y, lambd):
    """
    We solve the optimization problem:
    1/2*||b-x||**2 + sum_j(lambda_i*x_j) with csts: x_1>=x_2>=...>=x_k>0 and sum(x_i) = 1
    """
    # Construct the problem.
    n = y.shape[0]
    x = Variable(n)
    objective = Minimize(1./n*sum_squares(x - y) + sum_entries(np.diag(lambd)*x))
    #We reformulate the constrains as: x_i - x_j >= 0 i,j in [k-1] and x_k > 0
    constraints = [(x[:n-1]-x[1:])>=0, x[-1]>0, sum_entries(x)==1]
    prob = Problem(objective, constraints)
    # The optimal objective is returned by prob.solve().
    result = prob.solve(solver=CVXOPT)
    #We project on the probability simplex
    # The optimal value for x is stored in x.value.
    return np.array(x.value).reshape(len(x.value))

def pi_differentLambdas_estim_fista_conic(X, means, covars, pi, L, lambd):
    """
    We use FISTA to accelerate the convergence of the algorithm
    we project the next step of the gradient descent on the probability simplex
    """
    t_previous = 1
    pi_previous = np.copy(pi)
    xi = np.copy(pi_previous)
    # the number of iterations is given on FISTA paper,
    # we took ||pi_hat-pi_star||**2 = len(pi)**2
    fista_iter = int(np.sqrt(2*len(pi)**2 * L) // 1)
    for _ in range(min(500, fista_iter)):
        pi_next = ordered_optim_proj(xi - 1./(np.sqrt(X.shape[0])*L)*simple_gradient(X, means, covars, xi), lambd)
        t_next = (1. + np.sqrt(1 + 4 * t_previous**2)) / 2
        xi = pi_next + (t_previous - 1) / t_next * (pi_next - pi_previous)
        pi_previous = np.copy(pi_next)
    return pi_next

def algo_different_lambdas_penalities_conic(X, max_clusters, n_iter, L, alpha=0.001):
    """
    we inject in the gradient the penality, and project the estimate in the
    probability simplex
    """
    lambd = lambda_list_BH(max_clusters, alpha)
    # initialization of the algorithm
    g = GMM(n_components=max_clusters, covariance_type= "full")
    g.fit(X)
    # we order for slope
    print
    pi_estim, means_estim, covars_estim = map(list,zip(*(sorted(zip(g.weights_, g.means_, g.covars_))[::-1])))
    print pi_estim
    print "Init EM pi: ",pi_estim
    N = len(X)
    K = len(pi_estim)
    for it in range(n_iter):
        # We estimate pi according to the penalities lambdas given
        pi_estim = pi_differentLambdas_estim_fista_conic(X, means_estim, covars_estim, pi_estim, L, lambd)
        # we remove the clusters with probability = 0
        non_zero_elements = np.nonzero(pi_estim)[0]
        K = len(non_zero_elements)
        pi_estim = np.array([pi_estim[i] for i in non_zero_elements])
        means_estim = np.array([means_estim[i] for i in non_zero_elements])
        covars_estim = np.array([covars_estim[i] for i in non_zero_elements])
        lambd = np.array([lambd[i] for i in non_zero_elements])
        # we estimate the conditional probability P(z=j/X[i])
        tau = tau_estim(X, means_estim, covars_estim, pi_estim)
        # Means
        means_estim = np.array([(tau[:, k]*X.T).sum(axis=1)*1/(N*pi_estim[k]) for k in range(K)])
        # covars
        covars_temp = np.array(
                [covar_estim(X, means_estim[k], tau[:, k], pi_estim[k]) for k in range(K)])
        non_empty_covar_idx = check_zero_matrix(covars_temp)
        pi_estim = [pi_estim[j] for j in non_empty_covar_idx]
        means_estim = [means_estim[j] for j in non_empty_covar_idx]
        covars_estim = [covars_estim[j] for j in non_empty_covar_idx]
        lambd = [lambd[j] for j in non_empty_covar_idx]
        K = len(pi_estim)
        if it%10 == 0 :
            print "iteration ",it, "pi: ", pi_estim
    return pi_estim, means_estim, covars_estim, tau_estim

def view2Ddata(X):
    from plotly.offline import plot
    import plotly.graph_objs as go

    # Create random data with numpy
    import numpy as np

    N = 1000
    random_x = np.random.randn(N)
    random_y = np.random.randn(N)

    # Create a trace
    trace = go.Scatter(
        x=X[:,0],
        y=X[:,1],
        mode = 'markers'
    )

    data = [trace]

    # Plot and embed in ipython notebook!
    plot(data, filename='Plot')

def gm_params_generator(d, k, sparse_proba=None):
    """
    We generate centers in [-0.5, 0.5] and verify that they are separated enough
    """
    #  we scatter the unit square on k squares, the min distance is given by c/sqrt(k)
    min_center_dist = 0.1/np.sqrt(k)
    centers = [np.random.rand(1, d)[0]-0.5]
    for i in range(k-1):
        center = np.random.rand(1, d)[0]-0.5
        distances = np.linalg.norm(
            np.array(centers) - np.array(center),
            axis=1)
        while len(distances[distances < min_center_dist]) > 0:
            center = np.random.rand(1, d)[0]-0.5
            distances = np.linalg.norm(
                np.array(centers) - np.array(center),
                axis=1)
        centers.append(center)
    # if sparse_proba is set :
    #    generate covariance matrix with the possibility to set the sparsity on the precision matrix,
    # we multiply by 1/k^2 to avoid overlapping
    if sparse_proba==None:
        A = [random.rand(d,d) for _ in range(k)]
        cov = [1e-2/(k**2)*(np.diag(np.ones(d))+np.dot(a,a.transpose())) for a in A]
    else:
        cov = np.array([np.linalg.inv(make_sparse_spd_matrix(d, alpha=sparse_proba)) for _ in range(k)])
    p = np.random.randint(1000, size=(1, k))[0]
    weights = 1.0*p/p.sum()
    return weights, centers, cov

"""
# In[33]:

pi, means, covars = gm_params_generator(2,3)
X,_ = gaussian_mixture_sample(pi, means, covars, 1e4)


# In[36]:

view2Ddata(X)


# # simulations

# In[39]:

# methode SLOPE
# avec pi_i ordonnés
# avec 1/2*||b-x||**2 + sum_j(lambda_i*x_j) with csts: x_1>=x_2>=...>=x_k>0 and sum(x_i) = 1
#utilisation de CVXPY
pi_e, means_e, covars_e, _  = algo_different_lambdas_penalities_conic(X,max_clusters=5,n_iter=100, L=1e5, alpha=0.0001 )
print "real pi: ", pi
print "estimated pi: ", pi_e
print "real means", means
print "estimated means", means_e


# In[38]:

# ancienne methode avec les meme lambda_i lambda_i = alpha*sqrt(2*log(max_clusters/i))
# sans les pi_i ordonnés
pi_e_2, means_e_2, covars_e_2, _ = algo_different_lambdas_penalities_1(X,max_clusters=5,n_iter=100, L=1e4, alpha=1)
print "real pi: ", pi
print "estimated pi: ", pi_e_2
print "real means", means
print "estimated means", means_e_2



#La projection nous ramene sur le "l'angle" du simplex à la valeur 1/K
#En utilisant les lambda_BH_i donnés dans "SLOPE is adaptive to unknown sparsity and asymptotically minimax" (SU, Candes 2015)
#conseillé par Pierre B.
#lambda_i = alpha*sqrt(2*log(max_clusters/i))
a = np.array([ 0.2933247 ,  0.09657283,  0.24179129 , 0.10878973 , 0.25952145])
ordered_optim_proj(a,lambda_list_BH(len(a)))


# In[25]:

#en donnant un alpha faible
a = np.array([ 0.2933247 ,  0.09657283,  0.24179129 , 0.10878973 , 0.25952145])
ordered_optim_proj(a,lambda_list_BH(len(a),alpha=0.01))


# In[26]:

#En utilisant les lambda_BH_i donnés dans "SLOPE is adaptive to unknown sparsity and asymptotically minimax" (SU, Candes 2015)
#conseillé par Pierre B.
#lambda_i = alpha*sqrt(2*log(max_clusters/i))

print lambda_list_BH(5)
print lambda_list_BH(5, alpha=0.01)


# In[27]:

#test en dim 2
print ordered_optim_proj(np.array([0.5,1]),np.array([1,1])) #en dehors de la projection sur la "face" du simplex, on arrive a l'angle
print ordered_optim_proj(np.array([0,1]),np.array([1,1])) # pareil
print ordered_optim_proj(np.array([1,0.5]),np.array([1,1]))
print ordered_optim_proj(np.array([0.5,1]),np.array([10,1]))
# ci dessus le comportement est normal
print ordered_optim_proj(np.array([0.5, 1, 0.5]),lambda_list_BH(3, alpha=1))
print ordered_optim_proj(np.array([0.5,1, 0.5]),lambda_list_BH(3, alpha=0.01))
print ordered_optim_proj(np.array([0.5,1, 0.5]),lambda_list_BH(3, alpha=0.001))

"""

"""
Ci dessous on decompose la minimisation de :

1/2*||b-x||**2 + sum_j(lambda_i*x_j) avec les contraintes: x_1>=x_2>=...>=x_k>0 and sum(x_i) = 1

On va d'abord faire la minimisation avec les contraintes : x_1>=x_2>=...>=x_k>0
Puis faire une projection sur le simplex prob.
"""

def minimization(y, lambd):
    """
    We solve the optimization problem:
    1/2*||b-x||**2 + sum_j(lambda_i*x_j) with csts: x_1>=x_2>=...>=x_k>0 and sum(x_i) = 1
    """
    # Construct the problem.
    n = len(y)
    x = Variable(n)
    objective = Minimize(1./n*sum_squares(x - y) + sum_entries(np.diag(lambd)*x))
    #We reformulate the constrains as: x_i - x_j >= 0 i,j in [k-1] and x_k > 0
    constraints = [x[-1]>0] + [x[i]>=0 for i in range(n-1)]
    prob = Problem(objective, constraints)
    # The optimal objective is returned by prob.solve().
    result = prob.solve(solver=CVXOPT)
    #We project on the probability simplex
    # The optimal value for x is stored in x.value.
    return x.value
"""
y = np.array([ 0.2933247 ,  0.19657283,  0.24179129 , 0.10878973 , 0.25952145])
y_temp = minimization(y, lambda_list_BH(len(y), alpha=0.01))
print y_temp


# In[30]:

#On remarque
#On projette:
simplex_proj(y_temp)
"""