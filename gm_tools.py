import numpy as np
from numpy.random import multivariate_normal, multinomial
from sklearn.datasets import make_sparse_spd_matrix


def gaussian_mixture_sample(pi, centers, sigmas, N):
    """
    generate a gaussian mixture of size N and gives also labels
    :param pi:
    :param centers:
    :param sigmas:
    :param N:
    :return: points, labels
    """
    rep = multinomial(N, pi, size=1)[0]
    d = len(centers[0])
    Z = np.zeros((N,d+1))
    c= []
    for i in range(len(rep)):
        c+=[i for _ in range(rep[i])]
    Z[:,d] = c
    c = []
    for k in range(len(rep)):
        c+=multivariate_normal(centers[k], sigmas[k], rep[k]).tolist()
    Z[:,range(d)] = c
    l = []
    np.random.shuffle(Z)
    return Z[:,range(d)], Z[:,d]


def gm_params_generator(d,k):
    centers =  np.random.randint(20, size=(k, d))-10
    cov = np.array([np.linalg.inv(make_sparse_spd_matrix(d, alpha=0.8)) for _ in range(k)])
    p = np.random.randint(1000, size=(1, k))[0]
    weights = 1.0*p/p.sum()
    return weights, centers, cov


def get_centers_order(real_centers, estim_centers):
    """
    :param real_centers:
    :param estim_centers:
    :return: list ok k clusters matched (REAL_centers, ESTIM_centers)
    """
    d = {}
    k = len(real_centers)
    for u in range(k):
        for v in range(k):
            d[(u,v)] = np.linalg.norm(real_centers[u]-estim_centers[v])
    mins = sorted(d, key=d.get)[:k] #Nice trick
    return mins

def match_labels(y_estim, clusters_match_list):
    for u,v in clusters_match_list:
        y_estim[y_estim==v] = u
    return y_estim

def clusters_stats(y_real, y_matched):
    couples = zip(y_real, y_matched)
    i=0
    for u,v in zip(y_real, y_matched):
        if u==v:
            i+=1
    return 1.0*i/len(y_matched)

def cluster_match(y_real, y_estim):
    """
    Same result as gets_center_order and match_labels
    :param y_real:
    :param y_estim:
    :return:
    """
    print set(y_real)
    print set(y_estim)
    y_real_segments = sorted([(len(y_real[y_real==k]),k) for k in set(y_real)], key=lambda x: x[1])
    y_estim_segments = sorted([(len(y_estim[y_estim==k]),k) for k in set(y_estim)], key=lambda x: x[1])
    return y_real_segments, y_estim_segments