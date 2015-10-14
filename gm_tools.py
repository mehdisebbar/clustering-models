import numpy as np
from numpy.random import multivariate_normal, multinomial
from sklearn.datasets import make_sparse_spd_matrix
from itertools import permutations

def gaussian_mixture_sample(pi, centers, sigmas, N):
    """
    generate a gaussian mixture of size N and gives also labels
    :param pi:
    :param centers:
    :param sigmas:
    :param N: sample size
    :return: points, labels
    """
    nb_of_clusters = len(pi)
    space_dimension = len(centers[0])
    sample_repartition_among_clusters = multinomial(N, pi, size=1)[0]
    Z = np.zeros((N, space_dimension+1))
    labels = []
    for i in range(nb_of_clusters):
        labels += [i for _ in range(sample_repartition_among_clusters[i])]
    Z[:, space_dimension] = labels
    samples = []
    for cluster_index in range(nb_of_clusters):
        samples += multivariate_normal(centers[cluster_index], sigmas[cluster_index], sample_repartition_among_clusters[cluster_index]).tolist()
    Z[:, range(space_dimension)] = samples
    np.random.shuffle(Z)
    return Z[:, range(space_dimension)], Z[:, space_dimension]

def gm_params_generator(d,k):
    #We generate centers and verify that they are separated enough
    centers = [20*np.random.rand(1, d)[0]-10]
    for i in range(k-1):
        center = 20*np.random.rand(1, d)[0]-10
        distances = np.linalg.norm(np.array(centers)-np.array(center), axis=1)
        while len(distances[distances<0.0]) > 0:
            center = 20*np.random.rand(1, d)[0]-10
            distances = np.linalg.norm(np.array(cnters)-np.array(center), axis=1)
        centers.append(center)

    cov = np.array([50*np.linalg.inv(make_sparse_spd_matrix(d, alpha=0.8)) for _ in range(k)])
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


def cont_val(y, y_estim, i, j):
    y_idx = []
    for idx,val in enumerate(y):
        if val == i:
            y_idx.append(idx)
    y_estim_idx = []
    for idx,val in enumerate(y_estim):
        if val == j:
            y_estim_idx.append(idx)
    return len(set(y_idx)&set(y_estim_idx))

def cont_matrix(y, y_estim, permut):
    k = len(permut)
    m = np.zeros((k,k))
    for i in range(k):
        for j in range(k):
            m[j,i]=cont_val(y, y_estim, i, permut[j])
    return m

def best_cont_matrix(y, y_estim):
    best_permut = list(set(y))
    best_diag_sum = 0
    for permut in permutations(set(y_estim)):
        diag_sum = sum(cont_matrix(y, y_estim, list(permut)).diagonal())
        if diag_sum > best_diag_sum:
            best_permut = list(permut)
            best_diag_sum = diag_sum
    return cont_matrix(y, y_estim, best_permut), best_permut, best_diag_sum



def match_labels(y_estim2, clusters_match_list):
    for u,v in clusters_match_list:
        y_estim2[y_estim2==v] = u
    return y_estim2

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
    print "%%%%%%%%%%%%%%%%%%"
    print set(y_estim)
    y_real_segments = sorted([(len(y_real[y_real==k]),k) for k in set(y_real)], key=lambda x: x[0])
    y_estim_segments = sorted([(len(y_estim[y_estim==k]),k) for k in set(y_estim)], key=lambda x: x[0])
    return y_real_segments, y_estim_segments