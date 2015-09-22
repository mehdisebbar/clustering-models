import cvxpy as cvx
import numpy as np
from numpy import linalg as LA
from sklearn.covariance import GraphLassoCV


def graphlasso(X, tauk, muk, lambd):  # tauk nxk matrix, muk: pxk
    Ss = []
    for k in range(len(muk)):
        Xc = X - muk[k]
        cluster_cov = np.array([np.array(np.mat(vect).T*np.mat(vect)) for vect in Xc]) #check perf with np.dot(X.T, X) and take diag
        Snk = np.dot(cluster_cov.T, tauk[:, k])
        S = cvx.semidefinite(len(muk[0]))
        obj = cvx.Maximize(cvx.log_det(S) - sum([(S*Snk)[i, i] for i in range(len(muk))]))
        constraints = [cvx.sum_entries(cvx.abs(S)) <= lambd[k]]
        # Form and solve optimization problem
        prob = cvx.Problem(obj, constraints)
        prob.solve()
        if prob.status != cvx.OPTIMAL:
            print "error"
            raise Exception('CVXPY Error')
        S = S.value
       # S[abs(S) <= 1e-4] = 0 #force exact zeros
        Ss += [S]
    return Ss

def gauss_dens_inv(mu, omega, x):
    return np.sqrt(np.linalg.det(omega) / ((2 * np.pi) ** len(x))) * np.exp(
        -0.5 * np.dot(np.dot((x.T - mu), omega), (x.T - mu).T))

def gaussM(pi, mus, sigmas, n):
    pi = np.random.multinomial(n, pi, size=1)[0]
    print "Repartition: ",pi
    S = np.random.multivariate_normal(mus[0], sigmas[0], pi[0])
    print pi
    for idx, nk in enumerate(pi[1:]):
        print idx
        S = np.vstack([S, np.random.multivariate_normal(mus[idx+1], sigmas[idx+1], nk)])
    np.random.shuffle(S)
    return S

def omega_sample(p):
    A = np.mat(np.random.randn(p, p))  # Unit normal gaussian distribution.
    # A[scipy.sparse.rand(p, p, 0.85).todense().nonzero()] = 0  # Sparsen the matrix.
    return A * A.T

def graphlassoEM(X, K, lambd):
    dim = len(X[0]) # dimension
    tau = np.random.rand(len(X), K)  # init tau_i_k
    a = np.random.random_sample(K)  # init pi
    pi = a / sum(a)  # pi
    mu = np.random.random_sample((K, dim))  # K centers of R^dim, BETWEEN  0-1
    omega = [omega_sample(dim) for _ in range(K)]
    for _ in range(30):  # iterations EM procedure
        for i in range(len(X)):  # estimation tau_i_k
            for k in range(K-1):  # for each cluster
                b = 0
                for j in range(K):
                    b += pi[j] * gauss_dens_inv(mu[j], omega[j], X[i])
                tau[i][k] = pi[k] * gauss_dens_inv(mu[k], omega[k], X[i]) / b
        tau[:,K-1] = 1.0 - np.sum(tau[:,:K-1], axis = 1)
        pi = 1.0 / len(X) * np.sum(tau, axis=0)
        for k in range(K): #estimates mu
            a = tau[:, k] #get column tau on k cluster
            c = (a * X.T).T #get tau_i_k * X
            mu[k] = 1.0 / (len(X) * pi[k]) * np.sum(c, axis=0)
        print mu
        omega = graphlasso(X, tau, mu, lambd)
    return omega, mu, pi
