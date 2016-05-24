#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.base import BaseEstimator
from sklearn.mixture import GMM

from tools.algorithms_benchmark import view2Ddata
from tools.gm_tools import gm_params_generator, gaussian_mixture_sample, covar_estim, score, tau_estim
from tools.matrix_tools import check_zero_matrix


class sqrt_lasso_gmm(BaseEstimator):
    def __init__(self, lambda_param=1, Lipshitz_c=1, n_iter=100, max_clusters=8, verbose=False):
        self.p = max_clusters
        self.n_iter = n_iter
        self.lambd = lambda_param
        self.L = Lipshitz_c
        self.verbose = verbose

    def get_params(self, deep=True):
        return {"max_clusters": self.p,
                "n_iter": self.n_iter,
                "lambda_param": self.lambd,
                "Lipshitz_c": self.L,
                "verbose": self.verbose}

    def fit(self, X, y=None):

        """
        We use a expectation/maximization algorithm with a lasso penalization on the weights vector
        """
        # initialization of the algorithm
        self.EPSILON = 1e-8
        self.fista_iter = 300
        g = GMM(n_components=self.p, covariance_type="full")
        g.fit(X)
        self.means_, self.covars_, self.pi_ = g.means_, g.covars_, g.weights_
        if self.verbose:
            print "Init EM pi: ", self.pi_
        self.N = len(X)
        K = len(self.pi_)
        for it in range(self.n_iter):
            # We estimate pi according to the penalities lambdas given
            self.pi_ = self.pi_sqrt_lasso_reduced_estim_fista(X, self.means_, self.covars_, self.pi_)
            # we remove the clusters with probability = 0
            non_zero_elements = np.nonzero(self.pi_)[0]
            K = len(non_zero_elements)
            self.pi_ = np.array([self.pi_[i] for i in non_zero_elements])
            self.means_ = np.array([self.means_[i] for i in non_zero_elements])
            self.covars_ = np.array([self.covars_[i] for i in non_zero_elements])
            # we estimate the conditional probability P(z=j/X[i])
            self.tau = tau_estim(X, self.means_, self.covars_, self.pi_)
            # Means
            self.means_ = np.array([(self.tau[:, k] * X.T).sum(axis=1) * 1 / (self.N * self.pi_[k]) for k in range(K)])
            # covars
            covars_temp = np.array(
                [covar_estim(X, self.means_[k], self.tau[:, k], self.pi_[k]) for k in range(K)])
            non_empty_covar_idx = check_zero_matrix(covars_temp)
            self.pi_ = [self.pi_[j] for j in non_empty_covar_idx]
            self.means_ = [self.means_[j] for j in non_empty_covar_idx]
            self.covars_ = [self.covars_[j] for j in non_empty_covar_idx]
            K = len(self.pi_)
            if self.verbose and it % 10 == 0:
                print "iteration ", it, "pi: ", self.pi_
        return self

    def pi_sqrt_lasso_reduced_estim_fista(self, X, means, covars, pi):
        """
        lasso with square root of pi estimation
        We use FISTA to accelerate the convergence of the algorithm
        we project the next step (squared) of the gradient descent on the unit circle
        """
        t_previous = 1
        # we delete the last element and take the square root of the vector
        alpha_previous = np.copy(np.sqrt(pi[:-1]))
        xi = np.copy(alpha_previous)
        # the number of iterations is given on FISTA paper,
        # we took ||pi_hat-pi_star||**2 = len(pi)**2
        # fista_iter = int(np.sqrt(2*len(pi)**2 * self.L) // 1)
        for _ in range(self.fista_iter):
            grad_step = xi - 1. / (np.sqrt(X.shape[0]) * self.L) * self.grad_sqrt_penalty(X, means, covars, xi)
            alpha_next = self.proj_unit_disk(grad_step)
            t_next = (1. + np.sqrt(1 + 4 * t_previous ** 2)) / 2
            xi = alpha_next + (t_previous - 1) / t_next * (alpha_next - alpha_previous)
            alpha_previous = np.copy(alpha_next)
        # We return the squared vector to obtain a probability vector sum = 1
        return np.append(alpha_next ** 2, max(0, 1 - np.linalg.norm(alpha_next) ** 2))

    def grad_sqrt_penalty(self, X, means, covars, alpha):
        """
        alpha is of dim p-1
        density is of dim p-1
        Evaluate the gradient of
        """
        dens_last_comp = multivariate_normal.pdf(X, means[len(alpha)], covars[len(alpha)]).reshape(X.shape[0], 1)
        dens_witht_p_comp = np.array(
            [multivariate_normal.pdf(X, means[i], covars[i]) for i in range(len(alpha))]).T - dens_last_comp
        # We reshape for the division and add EPSILON to avoid zero division
        # we add the lambda penality
        return self.lambd - 2. / X.shape[0] * (alpha * dens_witht_p_comp /
                                               (self.EPSILON + dens_last_comp + ((alpha ** 2) * dens_witht_p_comp).sum(
                                                   axis=1)
                                                .reshape(X.shape[0], 1))).sum(axis=0)

    def proj_unit_disk(self, v):
        """
        we receive a vector [v1,v2,...,vp] and project [v1,v2,...,vp-1] on the unit disk.
        """
        if np.linalg.norm(v) ** 2 <= 1:
            return v
        else:
            return v / np.linalg.norm(v)

    def score(self, X, y=None):
        """
        Loglikelihood of the model on the dataset X
        """
        return 1. / X.shape[0] * np.log((np.array([multivariate_normal.pdf(
            X, self.means_[i], self.covars_[i]) for i in range(len(self.pi_))]).T * self.pi_).sum(axis=1)).sum(axis=0)


if __name__ == '__main__':
    """
    a test
    """
    pi, means, covars = gm_params_generator(2, 3)
    means = [np.array([0, 0]), np.array([1, 1]), np.array([0, 2])]
    X, _ = gaussian_mixture_sample(pi, means, covars, 1e4)
    view2Ddata(X)
    # methode (square root) lasso
    # avec pi_i non ordonnés
    max_clusters = 5
    lambd = np.sqrt(2 * np.log(max_clusters) / X.shape[0])
    cl = sqrt_lasso_gmm(max_clusters=max_clusters, n_iter=100, Lipshitz_c=10, lambda_param=lambd, verbose=True)
    print lambd
    print "real pi: ", pi
    cl.fit(X)
    print "estimated pi: ", cl.pi_
    print "real means", means
    print "estimated means", cl.means_
    print "score: ", cl.score(X)
    print "score L*: ", score(X, pi, means, covars)
