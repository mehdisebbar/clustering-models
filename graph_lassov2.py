
from sklearn.base import BaseEstimator
from sklearn.utils import check_array
from sklearn.cluster import KMeans
import cvxpy as cvx
import numpy as np
from numpy import linalg as LA
from scipy.stats import multivariate_normal
class lassoEM(BaseEstimator):
    """

    """
    def __init__(self, n_components=1, n_iter=1, lambd = None):
        self.n_components = n_components
        self.n_iter = n_iter
        self.min_covar = 1e-3
        if lambd == None:
            self.alpha = [10 for _ in range(self.n_components)]
        else:
            self.alpha = lambd


    def fit(self, X, y=None):

        #Init step: (based on Kmeans)

        X = check_array(X, dtype=np.float64)
        if X.shape[0] < self.n_components:
            raise ValueError(
                'GMM estimation with %s components, but got only %s samples' %
                (self.n_components, X.shape[0]))

        max_log_prob = -np.infty
        self.centers_ = KMeans(n_clusters= self.n_components, init="k-means++").fit(X).cluster_centers_
        self.pi_ = np.tile(1.0 / self.n_components,self.n_components)

        cv = np.linalg.inv(np.cov(X.T) + self.min_covar * np.eye(X.shape[1]))
        if not cv.shape:
            cv.shape = (1, 1)
        self.omega_ = np.tile(cv, (self.n_components, 1, 1))

        N = len(X)
        Klist = range(self.n_components)
        p = len(X[0])

        current_log_likelihood = None

        for i in range(self.n_iter):

            prev_log_likelihood = current_log_likelihood
            #Expectation step
            tau = [[ self.tauik(x, k) for k in Klist] for x in X]

            #Maximization step

            rho = [[self.rhoik(tau, i, k) for k in Klist] for i in range(N)]

            self.centers_ = [[sum([rho[i][k]*X[i][j] for i in range(N)])
                              for j in range(len(X[0]))] for k in Klist]
            self.pi_ = [sum( [taui[k] for taui in tau ])/len(X) for k in Klist]

            sn = [[[sum([(X[i][j] - self.centers_[k][j]) * (X[i][l] - self.centers_[k][l]) * rho[i][k] for i in range(N)])
                  for l in range(p)]
                 for j in range(p)]
                for k in Klist]

            self.omega_ = [self.omega_solve(sn[k], self.alpha) for k in Klist]


    def tauik(self, x, k):
        return (self.pi_[k]*gauss_dens_inv(self.centers_[k], self.omega_[k], x))/\
               (sum([self.pi_[j]*gauss_dens_inv(self.centers_[j], self.omega_[j], x) for j in range(self.n_components)]))

    def rhoik(self, tau, i, k):
        return tau[i][k]/(sum([tauj[k] for tauj in tau]))

    def omega_solve(self, snk, alpha):

        # Create a variable that is constrained to the positive semidefinite cone.
        n = len(snk[0])
        S = cvx.semidefinite(n)

    # Form the logdet(S) - tr(SY) objective. Note the use of a set
    # comprehension to form a set of the diagonal elements of S*Y, and the
    # native sum function, which is compatible with cvxpy, to compute the trace.
    # TODO: If a cvxpy trace operator becomes available, use it!
        obj = cvx.Minimize(-cvx.log_det(S) + sum([(S*np.array(snk))[i, i] for i in range(n)]))

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
        return np.array(S).tolist()

    def loglikelihood(self, X, ):
        pass

def gauss_dens_inv(mu, omega, x):
    return np.sqrt(np.linalg.det(omega) / ((2 * np.pi) ** len(x))) * np.exp(
        -0.5 * np.dot(np.dot((x.T - mu), omega), (x.T - mu).T))

def gaussM(pi, mus, sigmas, n):
    pi = np.random.multinomial(n, pi, size=1)[0]
    print "Repartition: ",pi
    S = np.random.multivariate_normal(mus[0], sigmas[0], pi[0])
    for idx, nk in enumerate(pi[1:]):
        S = np.vstack([S, np.random.multivariate_normal(mus[idx+1], sigmas[idx+1], nk)])
    np.random.shuffle(S)
    return S


def gaussM_1d(x, pi, mus, sigmas):
    d = sum([pi[k]*multivariate_normal(mus[k], sigmas[k]).pdf(x) for k in range(len(pi))]).sum()
    return d
def gaussM_Md(X, pi, mus, sigmas):
    return [gaussM_1d(x, pi, mus, sigmas) for x in X]


if __name__ == '__main__':

    weigths = [0.2, 0.8]
    centers = [[0, 0], [5, 5]]
    vars = [[[3, 1],[1, 1]], [[4,0],[0,2]]]


    X = gaussM(weigths,centers,vars, 1000)
    lasso = lassoEM(n_components=2, n_iter=10, lambd=20)
    lasso.fit(X)

    print "Lasso ==============="
    print "\ncovars "
    for var in lasso.omega_:
        print "precision matrix"
        for i in var:
            print i

    print "\nEstimated weights :"
    print lasso.pi_
    print "\nEstimated variances :"
    for var in lasso.omega_:
        print LA.inv(np.array(var))
    print "\nEstimated means :"
    for mean in lasso.centers_:
        print mean






