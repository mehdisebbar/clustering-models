import numpy as np
from numpy.random import multivariate_normal, multinomial
from sklearn.datasets import make_sparse_spd_matrix


def gaussian_mixture_sample(pi, centers, sigmas, N):
    rep = multinomial(N, pi, size=1)[0]
    Z = [multivariate_normal(centers[k], sigmas[k], rep[k]) for k in range(len(rep))]
    l = []
    for mixture in Z:
        for el in mixture:
            l.append(el)
    X = np.array(l)
    np.random.shuffle(X)
    return X


def gm_params_generator(d,k):
    centers =  np.random.randint(20, size=(k, d))-10
    cov = np.array([np.linalg.inv(make_sparse_spd_matrix(d, alpha=0.8)) for _ in range(k)])
    p = np.random.randint(1000, size=(1, k))[0]
    weights = 1.0*p/p.sum()
    return weights, centers, cov
