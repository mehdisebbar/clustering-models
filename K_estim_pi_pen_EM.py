import numpy as np
from cvxpy import *
from scipy.stats import multivariate_normal
from sklearn.base import BaseEstimator
from sklearn.mixture import GMM
from sklearn.utils import check_array


class GraphLassoMix(BaseEstimator):
    """
    EM algorithm with an estimation of number of clusters
    """
    def __init__(self, n_iter=10):
        self.K = 5
        self.n_iter = n_iter
        self.fista_iter=10
        self.lambd_pi_pen = 10
        self.lambd_fista = 1e-3
        self.L = 1e3 #Lipschitz constant of grad

    def fit(self, X, y=None):
        """
        :param X:
        :param y:
        :return:
        """
        X = check_array(X, dtype=np.float64)

        #init with EM algorithm:
        g = GMM(n_components=self.K)
        g.fit(X)
        means, covars, pi = g.means_, g.covars_, g.weights_
        self.N = len(X)

        # begin Iteration procedure, we estimates the weights (pi) with a penalization, then Taum means, covars,
        # with the explicit solution from EM

        for j in range(self.n_iter):
            print "Algo Iteration: ",j
            pi = self.pi_estim(X, means, covars, pi)
            print pi
            tau = self.tau(X, means, covars, pi)
            means = np.array([(tau[:,k]*X.T).sum(axis=1)*1/(self.N*pi[k]) for k in range(self.K)])
            covars = np.array([self.covar(X, means[k], tau[:, k], pi[k]) for k in range(self.K)])
        return pi, tau, means, covars


    def gradient(self,X, means, covars, pi_reduced):
        """
        calcule le gradient
        peut etre changer les arguments et faire passer lambd_pi
        :param X:
        :param means: vecteur de moyennes (K centres)
        :param covars: K covars
        :param pi_reduced: pi de dim K-1
        :return:
        """
        k = self.K-1
        grad =  -np.array(
            [
                (multivariate_normal.pdf(X, means[l], covars[l])- multivariate_normal.pdf(X, means[k], covars[k])
              )/(
                    multivariate_normal.pdf(X, means[k], covars[k]) + np.array(
                        [ pi_reduced[j]*(multivariate_normal.pdf(X, means[j], covars[j]) - multivariate_normal.pdf(X, means[k], covars[k])) for j in range(k)]
                    ).T.sum(axis=1)
                ) for l in range(k)]).sum(axis=1) + self.lambd_pi_pen
        return grad

    def pi_estim(self, X, means, covars, weights):
        """
        Fista algorithm, estimates pi with a penalization of the k-1 values
        :param X:
        :param means:
        :param covars:
        :param pi_previous: a K-dim weight vector
        :param K:
        :param lambd_fista:
        :return: weights, a K-dim vector of sum = 1
        """

        #Init
        pi_previous = np.copy(weights[:-1])
        xi_previous = np.copy(pi_previous)
        pi = Variable(self.K-1)
        constraints = [sum_entries(pi) <= 1, pi>= 0]

        #iterations FISTA
        t_previous = 1
        for i in range(self.fista_iter):
            prob = Problem(Minimize(self.L/2*norm(pi - (xi_previous - 1/self.L * self.gradient(X, means, covars, xi_previous)))**2), constraints)
            prob.solve(solver=MOSEK)
            pi_next = np.array([x[0] for x in np.array(pi.value)])
            t_next = (1. + np.sqrt(1+4*t_previous**2))/2
            xi_previous = pi_next + (t_previous - 1)/t_next * (pi_next - pi_previous)
            #TODO checking pi
            #pi_next = np.array([max(0,pi_j) for pi_j in pi_next])
        print prob.status
        return np.array([x for x in pi_next]+[1-pi_next.sum()])

    def tau(self, X, centers, covars, pi):
        densities = np.array([multivariate_normal.pdf(X, centers[k], covars[k]) for k in range(len(pi))]).T * pi
        s = densities.sum(axis=1)
        return (densities.T/(densities.sum(axis=1))).T


    def covar(self, X, mean, tau, pi):
        """
        emp covariance of EM
        :param mean: mean for one cluster
        :param pi: pi for this cluster
        :param N: lenth of X
        :param tau: vector of proba for each X[i] in the cluster, given by tau[:,k]
        :return: emp covariance matrix of this cluster
        """
        N = len(X)
        Z= np.sqrt(tau).reshape(N,1)*(X-mean)
        return 1/(pi*N)*Z.T.dot(Z)