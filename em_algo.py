import numpy as np
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator
from scipy.stats import multivariate_normal

class EM(BaseEstimator):
    
    def __init__(self, kmax=2, n_iter=10):
        self.kmax = kmax
        self.n_iter = n_iter

    def fit(self, X):
        #init
        self.N = X.shape[0]
        kmeans = KMeans(self.kmax)
        kmeans.fit(X)
        self.centers = kmeans.cluster_centers_
        self.pi = np.array([1.0*len(kmeans.labels_[kmeans.labels_==i])/self.N for i in range(self.kmax)])
        self.covars = np.array([np.cov((X[kmeans.labels_==i]- kmeans.cluster_centers_[i]).T) for i in range(self.kmax)])
        self.tau = self.tau_gen(X)
        #algorithm starts
        for _ in range(self.n_iter):
            print "expect"
            self.expectation(X)
            print "max"
            self.maximization(X)
        return self.pi, self.centers, self.covars
    
    def tau_gen(self, X):
        densities = np.array([multivariate_normal(self.centers[k], self.covars[k]).pdf(X) for k in range(self.kmax)]).T*self.pi
        s = densities.sum(axis=1)
        return (densities.T/(densities.sum(axis=1))).T
    
    def covar_gen(self, X, i):
        a = (X-self.centers[i])*(np.sqrt(self.tau[:,i]).reshape(-1,1))
        return a.T.dot(a)/(self.N*self.pi[i])
        
    def expectation(self, X):
        return self.tau_gen(X)
    
    def maximization(self, X):
        self.pi = self.tau.sum(axis=0)/self.N
        self.centers = np.array([(X*(self.tau[:,i].reshape(-1,1))).sum(axis=0)/(self.N*self.pi[i]) for i in range(self.kmax)])
        self.covars = np.array([self.covar_gen(X, i) for i in range(self.kmax)])
        