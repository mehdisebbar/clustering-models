from itertools import permutations
from numba import jit
import numpy as np
import scipy.stats
from numpy.random import multivariate_normal, multinomial
from scipy import random
from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_sparse_spd_matrix

EPSILON = 1e-100

class DatasetGenTool(object):
    """
    gen tool
    """
    def __init__(self, d, k):
        self.d = d
        self.k = k

    def generate(self, size, params = None, test_train_ratio = 0.3):
        """
        Generate a gaussian mix dataset with train and test sets of a given size
        params is a dict with the keys pi, centers and covars
        returns self.X_train, self.X_test, self.Y_train, self.Y_test
        """
        if params == None:
            print("No parameters given, generating dataset with random parameters")
            self.pi_real, self.centers_real, self.covars_real =\
            gm_params_generator(self.d, self.k)
        else:
            self.pi_real, self.centers_real, self.covars_real =\
            params["pi"], params["centers"], params["covars"]
        #We generate the data and split into 2 sets, train and test
        self.X, self.Y = gaussian_mixture_sample(self.pi_real,
                                                 self.centers_real,
                                                 self.covars_real,
                                                 size)
        self.X_train, self.X_test, self.Y_train, self.Y_test =\
        train_test_split(self.X,self.Y, test_size = test_train_ratio)
        return self.X_train, self.X_test, self.Y_train, self.Y_test

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
    Z = np.zeros([int(N), int(space_dimension + 1)])
    labels = []
    for i in range(nb_of_clusters):
        labels += [i for _ in range(sample_repartition_among_clusters[i])]
    Z[:, space_dimension] = labels
    samples = []
    for cluster_index in range(nb_of_clusters):
        samples += multivariate_normal(
            centers[cluster_index],
            sigmas[cluster_index],
            sample_repartition_among_clusters[cluster_index]
        ).tolist()
    Z[:, range(space_dimension)] = samples
    np.random.shuffle(Z)
    return Z[:, range(space_dimension)], Z[:, space_dimension]


def gm_params_generator(d, k, sparse_proba=None, alpha=5, min_center_dist=None):
    """
    We generate centers in [-0.5, 0.5] and verify that they are separated enough
    alpha is the size of the grid
    """
    #  we scatter the unit square on k squares, the min distance is given by alpha/sqrt(k)
    if min_center_dist == None:
        min_center_dist = alpha / np.sqrt(k)
    centers = [alpha*(np.random.rand(1, d)[0]-0.5)]
    for i in range(k-1):
        center = alpha*(np.random.rand(1, d)[0]-0.5)
        distances = np.linalg.norm(
            np.array(centers) - np.array(center),
            axis=1)
        while len(distances[distances < min_center_dist]) > 0:
            center = alpha*(np.random.rand(1, d)[0]-0.5)
            distances = np.linalg.norm(
                np.array(centers) - np.array(center),
                axis=1)
        centers.append(center)
    # if sparse_proba is set :
    #    generate covariance matrix with the possibility to set the sparsity on the precision matrix,
    # we multiply by 1/k^2 to avoid overlapping
    if sparse_proba == None:
        A = [random.rand(d, d) for _ in range(k)]
        cov = [alpha * 1e-2 / (k ** 2) * (np.diag(np.ones(d)) + np.dot(a, a.transpose())) for a in A]
    else:
        cov = np.array([np.linalg.inv(make_sparse_spd_matrix(d, alpha=sparse_proba)) for _ in range(k)])
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
            d[(u, v)] = np.linalg.norm(real_centers[u]-estim_centers[v])
    mins = sorted(d, key=d.get)[:k]  # Nice trick
    return mins

@jit
def cont_val(y, y_estim, i, j):
    y_idx = []
    for idx, val in enumerate(y):
        if val == i:
            y_idx.append(idx)
    y_estim_idx = []
    for idx, val in enumerate(y_estim):
        if val == j:
            y_estim_idx.append(idx)
    return len(set(y_idx) & set(y_estim_idx))

@jit
def cont_matrix(y, y_estim, permut):
    k = len(permut)
    m = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            m[j, i] = cont_val(y, y_estim, i, permut[j])
    return m

@jit
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
    for u, v in clusters_match_list:
        y_estim2[y_estim2 == v] = u
    return y_estim2


def clusters_stats(y_real, y_matched):
    i = 0
    for u, v in zip(y_real, y_matched):
        if u == v:
            i += 1
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
    y_real_segments = sorted([(len(y_real[y_real == k]),
                               k) for k in set(y_real)], key=lambda x: x[0])
    y_estim_segments = sorted([(len(y_estim[y_estim == k]),
                                k) for k in set(y_estim)], key=lambda x: x[0])
    return y_real_segments, y_estim_segments


def gauss_mix_density(x, pi, means, covars):
    return np.array([pi[j] * scipy.stats.multivariate_normal.pdf(
        x, means[j], covars[j]) for j in range(len(pi))]).sum()


def tau_estim(X, centers, covars, pi):
    """
    return the conditional probability matrix P(z=k/X_i) for a gaussian mixture. Expectation step in EM
    """
    from scipy.stats import multivariate_normal
    try:
        densities = np.array(
            [multivariate_normal.pdf(X, centers[k], covars[k], allow_singular=True) for k in range(len(pi))]).T * pi
        return (densities.T / (EPSILON + densities.sum(axis=1))).T
    except np.linalg.LinAlgError as e:
        print "Error on density computation for tau", e


def covar_estim(X, mean, tau, pi):
    """
    empirical covariance matrix of EM
    :param mean: mean for one cluster
    :param pi: pi for this cluster
    :param N: lenth of X
    :param tau: vector of proba for each X[i] in the cluster, given by tau[:,k]
    :return: emp covariance matrix of this cluster
    """
    N = len(X)
    Z = np.sqrt(tau).reshape(N, 1) * (X - mean)
    return 1 / (pi * N) * Z.T.dot(Z)


def score(X, weights, means, covars):
    """
    Loglikelihood of the GM model with param (weights, centers, covars) on the dataset X
    """
    from scipy.stats import multivariate_normal
    return np.log(EPSILON + (np.array([multivariate_normal.pdf(
        X, means[i], covars[i]) for i in range(len(weights))]).T * weights).sum(axis=1)).sum(axis=0)
