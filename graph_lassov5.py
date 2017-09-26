
from sklearn.base import BaseEstimator
from sklearn.utils import check_array
import numpy as np
from sklearn.cluster import KMeans
from sklearn.covariance import GraphLasso
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr
rglasso = importr('glasso')
rpy2.robjects.numpy2ri.activate()
from numba import jit

class GraphLassoMix(BaseEstimator):
    """

    """
    def __init__(self, n_components=2, n_iter=20, alpha = None):
        self.n_components = n_components
        self.n_iter = n_iter
        self.min_covar = 1e-3
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
            #print "Beginning Step: ",i

        #Expectation Step
            self.t = self.tau(X, self.N, self.p, self.n_components)

        #Maximization Step
            self.pi = self.t.sum(axis=0)*1/self.N
            self.centers = np.array([(self.t[:,k]*X.T).sum(axis=1)*1/(self.N*self.pi[k]) for k in range(self.n_components)])
            #We normalize X with tau for sklearn graphlasso estimator
            Z = self.build_Z(X, self.N, self.p, self.centers, self.t, self.n_components)
            self.omegas = np.array([self.model[k].fit(Z[k]).precision_ for k in range(self.n_components)])
            #self.omegas = np.array([ self.r_lasso_wrapper(Z[k]) for k in range(self.n_components)])
        self.clusters_assigned = self.t.argmax(axis = 1)
    
    @jit
    def cluster_assigned(self, X, N, p, n_components):
        res = np.zeros([n_components, N, p])
        for k in range(range(len(self.pi))):
            res[k]=self.gauss_dens_inv_all(X, self.centers[k], self.omegas[k])
        return res.argmax(axis = 1)

    @jit
    def build_Z(self, X, N, p, centers, t, n_components):
        res = np.zeros([n_components, N, p])
        for k in range(n_components):
            res[k]=((X-centers[k]).T*np.sqrt(N*t[:,k]/t[:,k].sum())).T
        return res
    
    @jit
    def gauss_dens_inv_all(self, X, center, omega):
        pi = np.pi
        return np.sqrt(np.linalg.det(omega) / ((2 * pi) ** self.p)) * \
               np.exp(-0.5*(np.dot(X-center,omega)*(X-center)).sum(axis =1))
    @jit
    def tau(self, X, N, p, n_components):
        densities = np.zeros([n_components, N])
        for k in range(n_components):
            densities[k] = self.gauss_dens_inv_all(X, self.centers[k], self.omegas[k])
        densities2 = densities.T*self.pi
        return (densities2.T/(densities2.sum(axis=1))).T

    def r_lasso_wrapper(self, Z_k):
        return np.array(rglasso.glasso(np.cov(Z_k.T),0.1)[1])
