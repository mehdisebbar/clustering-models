#!/usr/bin/env python
# -*- coding: utf-8 -*-

from functools import partial

import numpy as np
from numba import jit
from scipy.stats import multivariate_normal
from scipy.stats import threshold
from sklearn.base import BaseEstimator
from sklearn.grid_search import GridSearchCV
from sklearn.mixture import GMM

import apgpy as apg
from grad_descent_algs import nmapg_linesearch
from tools.gm_tools import gm_params_generator, gaussian_mixture_sample, covar_estim, score, tau_estim
from tools.math import bic_scorer
from tools.math import proj_unit_disk
from tools.matrix_tools import check_zero_matrix, clean_nans
from tools.matrix_tools import weights_compare

class sqrt_lasso_gmm(BaseEstimator):
    def __init__(self, lambd=1, lipz_c=1, n_iter=200, fista_iter=500, max_clusters=8, verbose=False, eps_stop=1e-30,
                 pen_power=2):
        self.max_clusters = max_clusters
        self.n_iter = n_iter
        self.lambd = lambd
        self.lipz_c = lipz_c
        self.verbose = verbose
        self.fista_iter = fista_iter
        self.eps_stop = eps_stop
        self.max_iter = n_iter
        self.pen_power = pen_power


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
        self.EPSILON = 1e-200
        g = GMM(n_components=self.max_clusters, covariance_type="diag")
        g.fit(X)
        self.means_, self.covars_, self.weights_ = g.means_, g.covars_, g.weights_
        self.N = len(X)
        self.X = X
        K = len(self.weights_)
        self.means_prev_ = []
        it = 0
        while not self.stopping_crit(self.means_, self.means_prev_, self.eps_stop) and it < self.max_iter:
            if len(self.weights_) == 1:
                return self
            # We estimate pi according to the penalities lambdas given
            self.means_ = clean_nans(self.means_)
            self.weights_ = clean_nans(self.weights_)
            self.means_prev_ = self.means_
            # self.weights_ = self.pi_sqrt_lasso_reduced_estim_fista(X,
            #                                                       self.means_, self.covars_, self.weights_
            #                                                       , self.fista_iter, self.eps_stop,
            #                                                       self.pen_power, self.lipz_c)
            self.weights_ = self.fista_backt(X, self.weights_)
            # we remove the clusters with probability = 0
            self.weights_ = threshold(self.weights_, threshmin=1e-40, newval=0)
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
            it += 1
        # Only for the bic scorer in cross_validation
        return self

    def stopping_crit(self, x, x_prev, eps):
        if len(x) == len(x_prev):
            return np.linalg.norm(np.array(x) - np.array(x_prev)) < eps
        return False

    def _n_parameters(self):
        # return len(self.weights_) * len(self.means_[0]) ** 2 + self.N * len(self.means_) + len(self.weights_)
        p = len(self.means_[0])
        k = len(self.weights_)
        return k + k * p + k * p ^ 2

    @jit()
    def fista_backt(self, X, pi, L=1e-4, eta=1.5):
        """
        fista avec backtracking
        uniquement pour puissance 2
        :param X:
        :param pi:
        :param grad_f:
        :param g:
        :param F:
        :param L:
        :param eta:
        :return:
        """
        it = 0
        t_previous = 1
        t_next = 1
        alpha_next = np.copy(np.sqrt(pi[:-1]))
        alpha_previous = np.zeros(alpha_next.shape[0])
        xi = np.copy(alpha_next)
        while not (np.linalg.norm(alpha_next - alpha_previous)) < self.eps_stop and it < self.fista_iter:

            pl = proj_unit_disk(xi - 1. / (np.sqrt(X.shape[0]) * L) *
                                self.grad_sqrt_penalty(xi, X, self.means_, self.covars_, 2))
            ql = self.f_pi_pen(xi, X, self.means_, self.covars_) + \
                 np.dot((pl - xi).T, self.grad_sqrt_penalty(xi, X, self.means_, self.covars_, self.pen_power)) + \
                 L / 2 * np.linalg.norm(xi - pl) ** 2
            while self.f_pi_pen(pl, X, self.means_, self.covars_) > ql:
                L = L * eta
                pl = proj_unit_disk(xi - 1. / (np.sqrt(X.shape[0]) * L) *
                                    self.grad_sqrt_penalty(xi, X, self.means_, self.covars_, 2))
                ql = self.f_pi_pen(xi, X, self.means_, self.covars_) + \
                     np.dot((pl - xi).T, self.grad_sqrt_penalty(xi, X, self.means_, self.covars_, self.pen_power)) + \
                     L / 2 * np.linalg.norm(xi - pl) ** 2
            alpha_next, alpha_previous = proj_unit_disk(xi - 1. / (np.sqrt(X.shape[0]) * L) *
                                                        self.grad_sqrt_penalty(xi, X, self.means_, self.covars_,
                                                                               2)), alpha_next
            t_next, t_previous = (1. + np.sqrt(1 + 4 * t_previous ** 2)) / 2, t_next
            xi = alpha_next + (t_previous - 1) / t_next * (alpha_next - alpha_previous)
            it += 1
        return np.append(alpha_next ** self.pen_power, max(0, 1 - (alpha_next ** self.pen_power).sum()))

    def nmapg_linesearch_weights_estim(self, X, means, covars, pi):
        grad_f = partial(self.grad_sqrt_penalty, X=X, means=means, covars=covars)
        F = partial(self.f_pi_pen, X=X, means=means, covars=covars)
        w = np.copy(np.sqrt(pi[:-1]))
        res = nmapg_linesearch(w, F=F, g=proj_unit_disk, grad_f=grad_f, eta=0.8, delta=1e-5, rho=0.8)
        return np.append(res ** 2, max(0, 1 - np.linalg.norm(res) ** 2))

    # @jit()
    def pi_sqrt_lasso_reduced_estim_fista(self, X, means, covars, pi, fista_iter, eps_stop, pen_power, lipz_c):
        """
        lasso with square root of pi estimation
        We use FISTA to accelerate the convergence of the algorithm
        we project the next step (squared) of the gradient descent on the unit circle
        """
        t_previous = 1
        t_next = 1
        # we delete the last element and take the square root of the vector
        alpha_next = np.copy(np.sqrt(pi[:-1]))
        xi = np.copy(alpha_next)
        alpha_previous = np.zeros(alpha_next.shape[0])
        # the number of iterations is given on FISTA paper,
        # we took ||pi_hat-pi_star||**2 = len(pi)**2
        # fista_iter = int(np.sqrt(2*len(pi)**2 * self.L) // 1)
        it = 0
        while not (np.linalg.norm(alpha_next - alpha_previous)) < eps_stop and it < fista_iter:
            alpha_previous = np.copy(alpha_next)
            grad_step = xi - 1. / (np.sqrt(X.shape[0]) * lipz_c) * self.grad_sqrt_penalty(xi, X, means, covars,
                                                                                          pen_power)
            alpha_next = proj_unit_disk(grad_step)
            t_next, t_previous = (1. + np.sqrt(1 + 4 * t_previous ** 2)) / 2, t_next
            xi = alpha_next + (t_previous - 1) / t_next * (alpha_next - alpha_previous)
            it += 1
        # We return the squared vector to obtain a probability vector sum = 1
        return np.append(alpha_next ** pen_power, max(0, 1 - (alpha_next ** pen_power).sum()))

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
        c_next = self.f_pi_pen(X, means, covars, x_next)
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
            if self.f_pi_pen(X, means, covars, z_next) <= (c_next - delta * np.linalg.norm(z_next - y_next) ** 2):
                x_next = z_next
            else:
                v_next = self.proj_unit_disk(x_next - alpha_x * self.grad_sqrt_penalty(X, means, covars, x_next))
                x_next = z_next if self.f_pi_pen(X, means, covars, z_next) <= self.f_pi_pen(X, means, covars,
                                                                                            v_next) else v_next
            t_previous = np.copy(t_next)
            t_next = (np.sqrt(4 * t_previous ** 2 + 1) + 1) / 2
            q_previous = np.copy(q_next)
            q_next = eta * q_previous + 1
            c_next = (eta * q_previous * c_next + self.f_pi_pen(X, means, covars, x_next)) / q_next
        return np.append(x_next ** 2, max(0, 1 - np.linalg.norm(x_next) ** 2))

    @jit()
    def f_pi_pen(self, alpha, X, means, covars, lambd=1, gamma=2):
        """
        evaluate penalized loglikelihood for a given pi = alpha**gamma
        used in nmapg_pi_estim
        """
        k = len(alpha)
        dens_last_comp = multivariate_normal.pdf(X, means[k], covars[k])
        d = np.empty((k, X.shape[0]))
        for i in range(k):
            d[i] = multivariate_normal.pdf(X, means[i], covars[i]) - dens_last_comp
        dens_witht_p_comp = np.array(d).T
        # We reshape for the division and add EPSILON to avoid zero division
        # we add the lambda penalit
        res = - 1. / X.shape[0] * (
                np.log((alpha ** gamma * dens_witht_p_comp).sum(axis=1) + dens_last_comp.reshape(
                    X.shape[0]))).sum() + self.lambd * (alpha.sum())
        return res

    @jit()
    def grad_sqrt_penalty(self, alpha, X, means, covars, pen_power):
        """
        alpha is of dim p-1
        density is of dim p-1
        Evaluate the gradient of
        """
        k = len(alpha)
        dens_last_comp = multivariate_normal.pdf(X, means[k], covars[k])
        d = np.empty((k, X.shape[0]))
        for i in range(k):
            d[i] = multivariate_normal.pdf(X, means[i], covars[i]) - dens_last_comp
        dens_with_p_comp = np.array(d).T
        # We reshape for the division and add EPSILON to avoid zero division
        # we add the lambda penality
        num = pen_power * alpha ** (pen_power - 1) * dens_with_p_comp
        den = (self.EPSILON + dens_last_comp + ((alpha ** pen_power) * dens_with_p_comp).sum(axis=1)).reshape(
            X.shape[0], 1)
        return self.lambd - 1. / X.shape[0] * (num / den).sum(axis=0)

    def score(self, X, y=None):
        """
        Loglikelihood of the model on the dataset X
        """
        return np.log((np.array([multivariate_normal.pdf(
            X, self.means_[i], self.covars_[i]) for i in range(len(self.weights_))]).T * self.weights_).sum(
            axis=1)).sum(axis=0)


def main_cv_fold():
    """
    a test
    """

    pi, means, covars = gm_params_generator(2, 3, min_center_dist=0.3)
    X, _ = gaussian_mixture_sample(pi, means, covars, 1e3)
    # view2Ddata(X)
    # methode (square root) lasso
    # avec pi_i non ordonnés
    max_clusters = 5
    lambd = np.sqrt(2 * np.log(max_clusters) / X.shape[0])
    param = {"lambd": [1e-1 * lambd], "lipz_c": [1]}

    clf = GridSearchCV(
        estimator=sqrt_lasso_gmm(n_iter=200, max_clusters=max_clusters, verbose=False),
        param_grid=param,
        cv=3, n_jobs=1,
                       scoring=bic_scorer, error_score=-1e10)
    print lambd
    print "real pi: ", pi
    clf.fit(X)
    print "estimated pi: ", clf.best_estimator_.weights_
    print "real means", means
    print "estimated means", clf.best_estimator_.means_
    print "score: ", score(X, clf.best_estimator_.weights_, clf.best_estimator_.means_, clf.best_estimator_.covars_) / \
                     X.shape[0]
    print "score L*: ", score(X, pi, means, covars) / X.shape[0]
    print clf.best_estimator_._n_parameters()
    params_GMM = {"n_components": range(2, max_clusters + 1)}

    clf_gmm = GridSearchCV(GMM(covariance_type='full'), param_grid=params_GMM, cv=3, n_jobs=1,
                           scoring=bic_scorer, error_score=-1e4)
    clf_gmm.fit(X)
    print "gmm:", clf_gmm.best_estimator_.weights_
    print "score gmm: ", score(X, clf_gmm.best_estimator_.weights_, clf_gmm.best_estimator_.means_,
                               clf_gmm.best_estimator_.covars_) / X.shape[0]


def main_timing():
    from time import time
    a = time()
    pi, means, covars = gm_params_generator(2, 3, min_center_dist=0)
    X, _ = gaussian_mixture_sample(pi, means, covars, 1e3)
    max_clusters = 5
    lambd = np.sqrt(2 * np.log(max_clusters) / X.shape[0])
    clf = sqrt_lasso_gmm(n_iter=200, max_clusters=max_clusters, lambd=lambd)
    clf.fit(X)
    b = time()
    print b - a


def basic_main():
    # from test import weights_compare
    from time import time

    pi, means, covars = gm_params_generator(2, 3, min_center_dist=0)
    X, _ = gaussian_mixture_sample(pi, means, covars, 1e3)
    max_clusters = 25
    lambd = np.sqrt(2 * np.log(max_clusters) / X.shape[0])
    clf = sqrt_lasso_gmm(n_iter=300, max_clusters=max_clusters, verbose=True, lambd=lambd, eps_stop=1e-50,
                         pen_power=2, lipz_c=1)
    print "real pi: ", pi
    a = time()
    clf.fit(X)
    b = time()
    print "algo done"
    print clf.weights_
    print "erreur:", weights_compare(sorted(clf.weights_), sorted(pi))
    print b - a
    print "EM+BIC"
    a = time()

    params_GMM = {"n_components": range(2, max_clusters + 1)}
    clf_gmm = GridSearchCV(GMM(covariance_type='full'), param_grid=params_GMM, cv=3, n_jobs=1,
                           scoring=bic_scorer)
    clf_gmm.fit(X)
    b = time()

    print weights_compare(sorted(clf_gmm.best_estimator_.weights_), sorted(pi))
    print b - a

if __name__ == '__main__':
    basic_main()
