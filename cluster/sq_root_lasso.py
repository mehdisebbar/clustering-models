#!/usr/bin/env python
# -*- coding: utf-8 -*-

from functools import partial

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.base import BaseEstimator
from sklearn.mixture import GMM

import apgpy as apg
from grad_descent_algs import nmapg_linesearch
from tools.algorithms_benchmark import view2Ddata
from tools.gm_tools import gm_params_generator, gaussian_mixture_sample, covar_estim, score, tau_estim
from tools.math import proj_unit_disk
from tools.matrix_tools import check_zero_matrix


class sqrt_lasso_gmm(BaseEstimator):
    def __init__(self, lambd=1, lipz_c=1, n_iter=100, fista_iter=200, max_clusters=8, verbose=False):
        self.max_clusters = max_clusters
        self.n_iter = n_iter
        self.lambd = lambd
        self.lipz_c = lipz_c
        self.verbose = verbose
        self.fista_iter = fista_iter

    def get_params(self, deep=True):
        return {"max_clusters": self.max_clusters,
                "n_iter": self.n_iter,
                "lambd": self.lambd,
                "lipz_c": self.lipz_c,
                "verbose": self.verbose}

    def fit(self, X, y=None):

        """
        We use a expectation/maximization algorithm with a lasso penalization on the weights vector
        """
        # initialization of the algorithm
        self.EPSILON = 1e-10
        g = GMM(n_components=self.max_clusters, covariance_type="full")
        g.fit(X)
        self.means_, self.covars_, self.weights_ = g.means_, g.covars_, g.weights_
        if self.verbose:
            print "Init EM pi: ", self.weights_
        self.N = len(X)
        self.X = X
        K = len(self.weights_)
        for it in range(self.n_iter):
            # We estimate pi according to the penalities lambdas given
            self.weights_ = self.pi_sqrt_lasso_reduced_estim_fista(X, self.means_, self.covars_, self.weights_)
            # we remove the clusters with probability = 0
            non_zero_elements = np.nonzero(self.weights_)[0]
            K = len(non_zero_elements)
            self.weights_ = np.array([self.weights_[i] for i in non_zero_elements])
            self.means_ = np.array([self.means_[i] for i in non_zero_elements])
            self.covars_ = np.array([self.covars_[i] for i in non_zero_elements])
            # we estimate the conditional probability P(z=j/X[i])
            self.tau = tau_estim(X, self.means_, self.covars_, self.weights_)
            # Means
            self.means_ = np.array(
                [(self.tau[:, k] * X.T).sum(axis=1) * 1 / (self.N * self.weights_[k]) for k in range(K)])
            # covars
            covars_temp = np.array(
                [covar_estim(X, self.means_[k], self.tau[:, k], self.weights_[k]) for k in range(K)])
            non_empty_covar_idx = check_zero_matrix(covars_temp)
            self.weights_ = [self.weights_[j] for j in non_empty_covar_idx]
            self.means_ = [self.means_[j] for j in non_empty_covar_idx]
            self.covars_ = [self.covars_[j] for j in non_empty_covar_idx]
            K = len(self.weights_)
            if self.verbose and it % 10 == 0:
                print "iteration", it, " lambda:", self.lambd, " L: ", self.lipz_c, " pi: ", self.weights_
        # Only for the bic scorer in cross_validation
        return self

    def _n_parameters(self):
        # return len(self.weights_) * len(self.means_[0]) ** 2 + self.N * len(self.means_) + len(self.weights_)
        return len(self.weights_)

    def nmapg_linesearch_weights_estim(self, X, means, covars, pi):
        grad_f = partial(self.grad_sqrt_penalty, X=X, means=means, covars=covars)
        F = partial(self.f, X=X, means=means, covars=covars)
        w = np.copy(np.sqrt(pi[:-1]))
        res = nmapg_linesearch(w, F=F, g=proj_unit_disk, grad_f=grad_f, eta=0.8, delta=1e-5, rho=0.8)
        return np.append(res ** 2, max(0, 1 - np.linalg.norm(res) ** 2))


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
            grad_step = xi - 1. / (np.sqrt(X.shape[0]) * self.lipz_c) * self.grad_sqrt_penalty(xi, X, means, covars)
            alpha_next = proj_unit_disk(grad_step)
            t_next = (1. + np.sqrt(1 + 4 * t_previous ** 2)) / 2
            xi = alpha_next + (t_previous - 1) / t_next * (alpha_next - alpha_previous)
            alpha_previous = np.copy(alpha_next)
        # We return the squared vector to obtain a probability vector sum = 1
        return np.append(alpha_next ** 2, max(0, 1 - np.linalg.norm(alpha_next) ** 2))

    def apgpy_pi_estim(self, X, means, covars, pi):

        alpha_next = apg.solve(self.grad_sqrt_penalty, self.proj_unit_disk, np.sqrt(pi[:-1]), quiet=True)
        return np.append(alpha_next ** 2, max(0, 1 - np.linalg.norm(alpha_next) ** 2))

    def nmapg_pi_estim(self, X, means, covars, pi, alpha_x=0.1, alpha_y=0.1, eta=0.8, delta=0.5):
        """
        lasso with square root of pi estimation
        We use non monotone accelerated proximal gradient method to accelerate the convergence of the algorithm
        we project the next step (squared) of the gradient descent on the unit circle
        """
        x_previous = np.copy(np.sqrt(pi[:-1]))
        x_next = np.copy(x_previous)
        z_next = np.copy(x_previous)
        t_next, t_previous = 1., 0.
        c_next = self.f(X, means, covars, x_next)
        q_next = 1
        for i in range(self.fista_iter):
            y_next = x_next + t_previous / t_next * (z_next - x_next) + (t_previous - 1) / t_next * (
            x_next - x_previous)
            if len(np.isnan(y_next)[np.isnan(y_next)]):
                print "fista iter", i
                print "y_next", y_next
                print "x_next", x_next
                print "x_previous", x_previous
                return 0
            z_next = self.proj_unit_disk(
                y_next - alpha_y * self.grad_sqrt_penalty(X, means, covars, y_next))  # WOOOOOT TO CHECK
            x_previous = np.copy(x_next)
            if self.f(X, means, covars, z_next) <= (c_next - delta * np.linalg.norm(z_next - y_next) ** 2):
                x_next = z_next
            else:
                v_next = self.proj_unit_disk(x_next - alpha_x * self.grad_sqrt_penalty(X, means, covars, x_next))
                x_next = z_next if self.f(X, means, covars, z_next) <= self.f(X, means, covars, v_next) else v_next
            t_previous = np.copy(t_next)
            t_next = (np.sqrt(4 * t_previous ** 2 + 1) + 1) / 2
            q_previous = np.copy(q_next)
            q_next = eta * q_previous + 1
            c_next = (eta * q_previous * c_next + self.f(X, means, covars, x_next)) / q_next
        return np.append(x_next ** 2, max(0, 1 - np.linalg.norm(x_next) ** 2))

    def f(self, alpha, X=None, means=None, covars=None):
        """
        evaluate penalized loglikelihood for a given pi
        used in nmapg_pi_estim
        """
        dens_last_comp = multivariate_normal.pdf(X, means[len(alpha)], covars[len(alpha)])
        dens_witht_p_comp = np.array(
            [multivariate_normal.pdf(X, means[i], covars[i]) - dens_last_comp for i in range(len(alpha))]).T
        # We reshape for the division and add EPSILON to avoid zero division
        # we add the lambda penalit
        try:
            res = - 1. / X.shape[0] * (
                np.log((alpha ** 2 * dens_witht_p_comp).sum(axis=1) + dens_last_comp.reshape(
                    X.shape[0]))).sum() + self.lambd * (alpha.sum())
        except:
            print "error in evaluating the F"
            print alpha
        return res

    def grad_sqrt_penalty(self, alpha, X=None, means=None, covars=None):
        """
        alpha is of dim p-1
        density is of dim p-1
        Evaluate the gradient of
        """
        if X == None:
            X, means, covars = self.X, self.means_, self.covars_
        dens_last_comp = multivariate_normal.pdf(X, means[len(alpha)], covars[len(alpha)])
        dens_with_p_comp = np.array(
            [multivariate_normal.pdf(X, means[i], covars[i]) - dens_last_comp for i in range(len(alpha))]).T
        # We reshape for the division and add EPSILON to avoid zero division
        # we add the lambda penality
        num = 2 * alpha * dens_with_p_comp
        den = (self.EPSILON + dens_last_comp + ((alpha ** 2) * dens_with_p_comp).sum(axis=1)).reshape(X.shape[0], 1)
        return self.lambd - 1. / X.shape[0] * (num / den).sum(axis=0)

    def score(self, X, y=None):
        """
        Loglikelihood of the model on the dataset X
        """
        return np.log((np.array([multivariate_normal.pdf(
            X, self.means_[i], self.covars_[i]) for i in range(len(self.weights_))]).T * self.weights_).sum(
            axis=1)).sum(axis=0)


if __name__ == '__main__':
    """
    a test
    """
    pi, means, covars = gm_params_generator(2, 4, min_center_dist=0.1)
    X, _ = gaussian_mixture_sample(pi, means, covars, 1e4)
    view2Ddata(X)
    # methode (square root) lasso
    # avec pi_i non ordonnés
    max_clusters = 7
    lambd = np.sqrt(2 * np.log(max_clusters) / X.shape[0])
    cl = sqrt_lasso_gmm(max_clusters=max_clusters, n_iter=50, lipz_c=1, lambd=lambd, verbose=True, fista_iter=300)
    print lambd
    print "real pi: ", pi
    cl.fit(X)
    print "estimated pi: ", cl.weights_
    print "real means", means
    print "estimated means", cl.means_
    print "score: ", cl.score(X)
    print "score L*: ", score(X, pi, means, covars)
    print cl._n_parameters()
