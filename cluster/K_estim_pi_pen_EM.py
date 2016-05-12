import numpy as np
from scipy.stats import multivariate_normal
from sklearn.base import BaseEstimator
from sklearn.mixture import GMM
from sklearn.utils import check_array
from tools.matrix_tools import check_zero_matrix
from tools.math import simplex_proj


class GraphLassoMix(BaseEstimator):
    """
    EM algorithm with an estimation of number of clusters
    """

    def __init__(self, lambda_param, n_iter=10, max_clusters = 10):
        self.K = max_clusters
        self.n_iter = n_iter
        self.lambd_pi_pen = lambda_param
        self.L = 1e5  # Lipschitz constant of grad
        eps = 0.1
        self.fista_iter = int(np.sqrt(self.K * self.L * eps) // 1)
        np.set_printoptions(linewidth = 200)

    def fit(self, X, y=None):
        """
        :param X:
        :param y:
        :return:
        """
        X = check_array(X, dtype=np.float64)

        # init with EM algorithm:
        g = GMM(n_components=self.K, covariance_type= "full")
        g.fit(X)
        means, covars, pi = g.means_, g.covars_, g.weights_
        self.N = len(X)


        # begin Iteration procedure, we estimates the weights (pi) with a penalization, then Taum means, covars,
        # with the explicit solution from EM

        for j in range(self.n_iter):
            print "iteration "+ str(j)
            pi = self.pi_estim(X, means, covars, pi)
            # we remove clusters such that pi_j = 0
            non_zero_elements = np.nonzero(pi)[0]
            self.K = len(non_zero_elements)
            pi, means, covars = np.array([pi[i] for i in non_zero_elements]), np.array(
                [means[i] for i in non_zero_elements]), np.array([covars[i] for i in non_zero_elements])
            #estimate tau
            tau = self.tau(X, means, covars, pi)
            #estimate means, covar t+1
            means = np.array([(tau[:, k]*X.T).sum(axis=1)*1/(self.N*pi[k]) for k in range(self.K)])
            covars = np.array(
                [self.covar(X, means[k], tau[:, k], pi[k]) for k in range(self.K)])
            #Removing empty covar matrices
            pi = [pi[j] for j in check_zero_matrix(covars)]
            means = [means[j] for j in check_zero_matrix(covars)]
            covars = [covars[j] for j in check_zero_matrix(covars) ]
            self.K = len(pi)
        return pi, tau.argmax(axis=1), means, covars

    def gradient(self, X, means, covars, pi_estim):
        """
        calcule le gradient
        peut etre changer les arguments et faire passer lambd_pi
        :param X:
        :param means: vecteur de moyennes (K centres)
        :param covars: K covars :param pi_estim: pi de dim K
        :return:
        """
        try:
            grad = -np.array([
                multivariate_normal.pdf(X, means[l], covars[l], allow_singular=True) /
                (np.array(
                  [pi_estim[
                        j] * multivariate_normal.pdf(X, means[j], covars[j], allow_singular=True) for j in range(self.K)]
                ).T.sum(axis=1)
                ) + self.lambd_indi(l) for l in range(self.K)]).sum(axis=1)
            return grad
        except ZeroDivisionError as e:
            print e
            raise Exception(e)



    def lambd_indi(self, l):
        # Nous donne lambda pour tout l sauf le dernier = K
        if l != self.K - 1:
            return self.lambd_pi_pen
        else:
            return 0

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

        # Init
        pi_previous = np.copy(weights)
        xi = np.copy(pi_previous)

        # iterations FISTA
        print "Entering FISTA"
        t_previous = 1
        for i in range(self.fista_iter):
            try:
                pi_est = xi - 1 / self.L * self.gradient(X, means, covars, xi)
                pi_next = simplex_proj(pi_est)
            except np.linalg.LinAlgError as e:
                print "Error on the estimation of pi, skipping FISTA and returning previous pi, error", e
                raise Exception(e)

            t_next = (1. + np.sqrt(1 + 4 * t_previous**2)) / 2
            xi = pi_next + (t_previous - 1) / t_next * (pi_next - pi_previous)
            pi_previous = np.copy(pi_next)
        print "FISTA Done"
        return pi_next


    def tau(self, X, centers, covars, pi):
        try:
            densities = np.array([multivariate_normal.pdf(X, centers[k], covars[k], allow_singular=True) for k in range(len(pi))]).T * pi
            return (densities.T/(densities.sum(axis=1))).T
        except np.linalg.LinAlgError as e:
            print "Error on density computation for tau", e

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

