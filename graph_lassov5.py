
from sklearn.base import BaseEstimator
from sklearn.utils import check_array
import numpy as np
from sklearn.cluster import KMeans
from sklearn.covariance import GraphLassoCV


class GraphLassoMix(BaseEstimator):
    """

    """
    def __init__(self, n_components=2, n_iter=5, alpha = None):
        self.n_components = n_components
        self.n_iter = n_iter
        self.min_covar = 1e-3
        if alpha == None:
            self.alpha = [10 for _ in range(self.n_components)]
        else:
            self.alpha = alpha
        self.model = [GraphLassoCV() for k in range(self.n_components)]

    def fit(self, X, y=None):
        """
        :param X:
        :param y:
        :return:
        """
        X = check_array(X, dtype=np.float64)
        if X.shape[0] < self.n_components:
            raise ValueError(
                'GMM estimation with %s components, but got only %s samples' %
                 (self.n_components, X.shape[0]))

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
            t = self.tau(X, self.centers, self.omegas, self.pi)

        #Maximization Step
            self.pi = t.sum(axis=0)*1/self.N
            self.centers = np.array([(t[:,k]*X.T).sum(axis=1)*1/(self.N*self.pi[k]) for k in range(self.n_components)])
            #We normalize X with tau for sklearn graphlasso estimator
            Z = [((X-self.centers[k]).T*np.sqrt(t[:,k]/t[:,k].sum())).T for k in range(self.n_components)]
            self.omegas = np.array([1.0/self.N*self.model[k].fit(Z[k]).precision_ for k in range(self.n_components)])

    def gauss_dens_inv_all(self, X, center, omega):
        pi = np.pi
        return np.sqrt(np.linalg.det(omega) / ((2 * pi) ** self.p)) * \
               np.exp(-0.5*(np.dot(X-center,omega)*(X-center)).sum(axis =1))

    def tau(self, X, centers, omegas, pi):
        densities = np.array([self.gauss_dens_inv_all(X, centers[k], omegas[k]) for k in range(len(pi))]).T*pi
        s = densities.sum(axis=1)
        return (densities.T/(densities.sum(axis=1))).T
