import sys

import numpy as np
from cvxpy import *
from scipy.stats import multivariate_normal
from scipy.stats import threshold

from gm_tools import score


def simplex_proj(v):
    """
    Projection sur le probability simplex
    http://arxiv.org/pdf/1309.1541.pdf
    :return:
    """
    # for reshaping from matrix type
    y = np.array(v).reshape(len(v))
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

def ordored_optim_proj_prob_simplex(b):
    """
    We solve the optimization problem:
    1/2*||b-x||**2 with csts: x_1>=x_2>=...>=x_k>0 and sum(x_i) = 1
    """
    # Construct the problem.
    n = b.shape[0]
    x = Variable(n)
    objective = Minimize(1./2*sum_squares(x - b))
    #We reformulate the constrains as: x_i - x_j >= 0 i,j in [k-1] and x_k > 0
    constraints = [(x[:n-1]-x[1:])>=0, x[-1]>0]
    prob = Problem(objective, constraints)
    # The optimal objective is returned by prob.solve().
    result = prob.solve()
    #We project on the probability simplex
    # The optimal value for x is stored in x.value.
    return simplex_proj(x.value)

def gradient_different_lambdas(X, means, covars, pi, lambd, EPSILON=1e-8):
    """
    Evaluate the gradient of -sum_i^n( log( sum_j^K (pi_j * phi(mu_j, sigma_j)(X_i) ))) + sum_l^K (lambda_l*pi_l)
    """
    densities = np.array([multivariate_normal.pdf(X, means[i], covars[i]) for i in range(len(means))]).T
    #We reshape for the division and add EPSILON to avoid zero division
    #we add the lambda penality (SLOPE like)
    return -(densities/(((densities*pi).sum(axis=1)).reshape(X.shape[0],1) + EPSILON)).sum(axis=0) + lambd


def covar(X, mean, tau, pi):
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


def proj_unit_disk(w, t=None):
    """
    we receive a vector [v1,v2,...,vp] and project [v1,v2,...,vp-1] on the positive unit disk.
    """
    v = threshold(w, threshmin=0, newval=0)
    if np.linalg.norm(v) ** 2 <= 1:
        return v
    else:
        return v / np.linalg.norm(v)


def bic_scorer(estimator, X, y=None):
    # we define a bic scoring method for the grid search
    try:
        return (2 * score(X, estimator.weights_, estimator.means_, estimator.covars_) -
                estimator._n_parameters() * np.log(X.shape[0]))
    except:
        print "Unexpected scoring error:", sys.exc_info()[0]
        return -9 * 1e5
