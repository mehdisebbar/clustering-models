import numpy as np
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.covariance import GraphLasso
from sklearn.utils import check_array

rglasso = importr('glasso')
rpy2.robjects.numpy2ri.activate()
from cvxpy import *


class GraphLassoMix(BaseEstimator):
    """

    """
    def __init__(self, n_components, n_iter=5, alpha = None):
        self.n_components = n_components+2
        self.n_iter = n_iter
        self.min_covar = 1e-3
        self.tresh = 1e-3
        self.lambd = 0.5
        if alpha == None:
            self.alpha = [10 for _ in range(self.n_components)]
        else:
            self.alpha = alpha
        self.model = [GraphLasso(alpha=self.alpha[k], assume_centered=True, tol=1e-4, verbose=True) for k in range(self.n_components)]

    def fit(self, X, y=None):
        """
        :param X:
        :param y:
        :return:
        """
        X = check_array(X, dtype=np.float64)
        #if X.shape[0] < self.n_components:
        #    raise ValueError(
        #        'GMM estimation with %s components, but got only %s samples' %
        #         (self.n_components, X.shape[0]))

        #Init of params
        cv = np.linalg.inv(np.cov(X.T))
        if not cv.shape:
            cv.shape = (1, 1)
        self.omegas = np.tile(cv, (self.n_components, 1, 1))
        self.centers = KMeans(n_clusters= self.n_components, init="k-means++").fit(X).cluster_centers_
        self.pi = np.tile(1.0 / self.n_components,self.n_components)
        self.N, self.p = X.shape

        #EM-Iterations
        for i in range(self.n_iter):
            print "Beginning Step: ",i

        #Expectation Step
            self.tau_mat = self.expectation_k_estim(X, self.centers, self.omegas, self.pi, self.lambd)

        #Maximization Step
            self.pi = self.tau_mat.sum(axis=0)*1./self.N
            self.centers = np.array([(self.tau_mat[:,k]*X.T).sum(axis=1)*1/(self.N*self.pi[k]) for k in range(self.n_components)])
            #We normalize X with tau for sklearn graphlasso estimator
            Z = [((X-self.centers[k]).T*np.sqrt(self.N*self.tau_mat[:,k]/self.tau_mat[:,k].sum())).T for k in range(self.n_components)]
            #self.omegas = np.array([self.model[k].fit(Z[k]).precision_ for k in range(self.n_components)])
            self.omegas = np.array([ self.r_lasso_wrapper(Z[k]) for k in range(self.n_components)])
            self.shrink_clusters(self.tresh)

        #end of iterations
        self.clusters_assigned = self.tau_mat.argmax(axis = 1)

    def cluster_assigned(self, X):
        d = np.array([self.gauss_dens_inv_all(X, self.centers[k], self.omegas[k]) for k in range(len(self.pi))]).T
        return d.argmax(axis = 1)



    def gauss_dens_inv_all(self, X, center, omega):
        pi = np.pi
        return np.sqrt(np.linalg.det(omega) / ((2 * pi) ** self.p)) * \
               np.exp(-0.5*(np.dot(X-center,omega)*(X-center)).sum(axis =1))

    def tau(self, X, centers, omegas, pi):
        densities = np.array([self.gauss_dens_inv_all(X, centers[k], omegas[k]) for k in range(len(pi))]).T*pi
        s = densities.sum(axis=1)
        return (densities.T/(densities.sum(axis=1))).T

    def r_lasso_wrapper(self, Z_k):
        return np.array(rglasso.glasso(np.cov(Z_k.T),0.1)[1])

    def expectation_k_estim(self, X, centers, omegas, pi, lambd):
        """
        We estimate tau by penalizing the number of clusters, we use Fista
        :return:
        """
        #definition de tau
        tau = [[Variable() for _ in range(self.n_components) ] for _ in range(self.N)] #definition de la matrice tau de n*k variables
        tau_previous = self.tau(X, centers, omegas, pi)
        xi_previous = np.copy(tau_previous)
        constraints = [sum_entries(bmat(ligne_n)) == 1 for ligne_n in tau ]+[item >=0 for sublist in tau for item in sublist ]  #contraintes sur somme des lignes

        #iterations FISTA
        t_previous = 1
        for i in range(10):
            print "FISTA it: "+ str(i)
            print "computing gradient"
            grad_xi_previous = self.gradient(xi_previous, X, centers, omegas, pi)
            print "solving"
            prob = Problem(Minimize(norm(bmat(tau) - (xi_previous+lambd*grad_xi_previous) )**2), constraints)
            prob.solve(solver=SCS, use_indirect=True)
            tau_next = np.array([[x.value for x in line] for line in tau])
            t_next = (1. + np.sqrt(1+4*t_previous**2))/2
            xi_previous = tau_next + (t_previous - 1)/t_next * (tau_next - tau_previous)
            print tau_next.sum(axis=1)
            print tau_next
        return tau_next

    def gradient(self, xi, X, centers, omegas, pi ):
        #give gradient of f on xi
        return np.array([np.log(self.gauss_dens_inv_all(X, centers[k], omegas[k])) + np.log(pi[k]/xi[:,k]) -1 + xi[:,k]/np.linalg.norm(xi, axis=0)[k] for k in range(self.n_components)]).T

    def shrink_clusters(self, thresh):
        print self.tau_mat.sum(axis=0)
        pass
