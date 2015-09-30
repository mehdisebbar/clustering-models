
from sklearn.base import BaseEstimator
from sklearn.utils import check_array
import numpy as np
from sklearn.cluster import KMeans
from sklearn.covariance import GraphLassoCV
from numpy import linalg as LA
from graph_lassov3 import gaussM
from sklearn.datasets import make_sparse_spd_matrix
import cvxpy as cvx
import matplotlib.pyplot as plt



class Graph_lasso_Mix(BaseEstimator):
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
        self.Klist = range(self.n_components)
        X = check_array(X, dtype=np.float64)
        if X.shape[0] < self.n_components:
            raise ValueError(
                'GMM estimation with %s components, but got only %s samples' %
                 (self.n_components, X.shape[0]))
        self.centers = KMeans(n_clusters= self.n_components, init="k-means++").fit(X).cluster_centers_
        self.pi = np.tile(1.0 / self.n_components,self.n_components)

        cv = np.linalg.inv(np.cov(X.T) + self.min_covar * np.eye(X.shape[1]))
        if not cv.shape:
            cv.shape = (1, 1)
        self.omegas = np.tile(cv, (self.n_components, 1, 1))
        self.N, self.p = X.shape

        for i in range(self.n_iter):
            print "Beginning Step: ",i
        #Expectation Step
            t = self.tau(X, self.centers, self.omegas, self.pi)
        #Maximization Step
            self.pi = t.sum(axis=0)*1/self.N
            self.centers = np.array([(t[:,k]*X.T).sum(axis=1)*1/(self.N*self.pi[k]) for k in range(self.n_components)])
            #We normalize X with tau for sklearn graphlasso estimator
            Z = [((X-centers[k]).T*np.sqrt(t[:,k]/t[:,k].sum())).T for k in range(self.n_components)]
            Z = [check_array(Z[k]) for k in range(self.n_components)]

            #self.omegas = np.array([self.model[k].fit(Z[k]).precision_ for k in range(self.n_components)])
            self.omegas = np.array([self.omega_solve(Z[k],20)   for k in range(self.n_components)])


    def gauss_dens_inv_all(self, X, center, omega):
        return np.sqrt(np.linalg.det(omega) / ((2 * np.pi) ** self.p)) * \
               np.exp(-0.5*(np.dot(X-center,omega)*(X-center)).sum(axis =1))

    def tau(self, X, centers, omegas, pi):
        densities = np.array([self.gauss_dens_inv_all(X, centers[k], omegas[k]) for k in range(len(pi))]).T*pi
        s = densities.sum(axis=1)
        return (densities.T/densities.sum(axis=1)).T

    def omega_solve(self, Z, alpha):

        # Create a variable that is constrained to the positive semidefinite cone.
        n = len(Z[0])
        S = cvx.semidefinite(n)

    # Form the logdet(S) - tr(SY) objective. Note the use of a set
    # comprehension to form a set of the diagonal elements of S*Y, and the
    # native sum function, which is compatible with cvxpy, to compute the trace.
    # TODO: If a cvxpy trace operator becomes available, use it!
        obj = cvx.Minimize(-cvx.log_det(S) + sum([(S*np.cov(Z.T))[i, i] for i in range(n)]))

    # Set constraint.
        constraints = [cvx.sum_entries(cvx.abs(S)) <= alpha]

    # Form and solve optimization problem
        prob = cvx.Problem(obj, constraints)
        prob.solve()
        if prob.status != cvx.OPTIMAL:
            raise Exception('CVXPY Error')

        # If the covariance matrix R is desired, here is how it to create it.

        # Threshold S element values to enforce exact zeros:
        S = S.value

        S[abs(S) <= 1e-4] = 0
        return np.array(S)

def gm_params_gen(d,k):
    centers =  np.random.randint(20, size=(k, d))-10
    cov = np.array([np.linalg.inv(make_sparse_spd_matrix(d)) for _ in range(k)])
    p = np.random.randint(1000, size=(1, k))[0]
    weights = 1.0*p/p.sum()
    return weights, centers, cov

if __name__ == '__main__':
    d = 20
    k=3
    N= 10000

    weights, centers, cov = gm_params_gen(d,k)
    X = gaussM(weights, centers, cov, N)
    X.shape
    lasso = Graph_lasso_Mix(n_components=k, n_iter=1)
    lasso.fit(X)

#  print "Lasso ==============="
#  print "\ncovars "
#  for var in lasso.omegas:
#      print "precision matrix"
#      for i in var:
#          print i

#  print "\nEstimated weights :"
#  print lasso.pi
#  print "\nEstimated variances :"
#  for var in lasso.omegas:
#      print LA.inv(np.array(var))
#  print "\nEstimated means :"
#  for mean in lasso.centers:
#      print mean
# Plot the results
plt.figure(figsize=(10, 6))
plt.subplots_adjust(left=0.02, right=0.98)
vmax = .9 * lasso.omegas[0].max()
for i,omega in enumerate(lasso.omegas):
    ax = plt.subplot2grid((k, 2), (i,0))
    plt.imshow(np.ma.masked_equal(omega, 0),
               interpolation='nearest', vmin=-vmax, vmax=vmax,
               cmap=plt.cm.RdBu_r)
    plt.xticks(())
    plt.yticks(())
    plt.title('Graphlasso precision')
    ax.set_axis_bgcolor('.7')
for i,omega in enumerate(cov):
    ax = plt.subplot2grid((k, 2), (i,1))
    plt.imshow(np.ma.masked_equal(np.linalg.inv(omega), 0),
               interpolation='nearest', vmin=-vmax, vmax=vmax,
               cmap=plt.cm.RdBu_r)
    plt.xticks(())
    plt.yticks(())
    plt.title('True precision')
    ax.set_axis_bgcolor('.7')
plt.show()
