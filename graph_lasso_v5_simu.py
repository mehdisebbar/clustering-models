from datetime import datetime
import os
import pickle
from time import time
import uuid
import numpy as np
from sklearn.mixture import GaussianMixture
from graph_lassov5 import GraphLassoMix
from tools.gm_tools_old import gaussian_mixture_sample, gm_params_generator, best_cont_matrix

FOLDER = "dg_"+str(datetime.now()).split(".")[0].replace(" ", "_").replace(":", ".") + "/"
os.makedirs(FOLDER)

def center_gen(dim, k):
    #generate centers in a grid 1
    centers = []
    for i in range(k):
        center_id = [int(x) for x in bin(i).split("b")[1]]
        complement = [0 for _ in range(dim-len(center_id))]
        centers.append(complement+center_id)
    return np.array(centers)

def main(d,k,N):
    try:
        #_ , centers, _ = gm_params_generator(d, k)
        centers = center_gen(d,k)
        weights = 1./k*np.ones(k)
        cov = 1e-3*np.array([np.diag(np.ones(d)) for _ in range(k)])
        X, Y = gaussian_mixture_sample(weights, centers, cov, N)
        lasso = GraphLassoMix(n_components=k, n_iter=30)
        t1_lasso = time()
        lasso.fit(X)
        t2_lasso=time()
        y_lasso = lasso.clusters_assigned
        gmm = GaussianMixture(n_components=k, covariance_type="full")
        t1_em= time()
        gmm.fit(X)
        y_em = gmm.predict(X)
        t2_em = time()
        #Recovering mapping
        permut_lasso = algo_score(Y, y_lasso)
        permut_gmm = algo_score(Y, y_em)
        #computing errors
        l = []
        for idx, val in enumerate(permut_lasso):
            l.append(1. / (cov[idx].shape[0] ** 2) * np.linalg.norm(np.linalg.inv(cov[idx]) - lasso.omegas[val]))
            #
        l2 = []
        for idx, val in enumerate(permut_gmm):
            l2.append(1. / (cov[idx].shape[0] ** 2) * np.linalg.norm(np.linalg.inv(cov[idx]) - np.linalg.inv(gmm.covariances_[val])))

        #print "OK, writing results"
        pickle.dump({"K" : k,
                             "p" : d,
                             "N" : N,
                             "time_em" : t2_em-t1_em,
                             "time_lasso" : t2_lasso - t1_lasso,
                             "error_lasso" : l,
                             "error_em" :l2,
                             "X" : X
                         }, open(FOLDER +
                                 "res_graph_lasso_" + "K" + str(k) + "p" + str(d) + "N" + str(N) +"_"+str(uuid.uuid4()), "wb"))
    except:
        return 0
def algo_score(Y, y_estim, t=0):
    mat, permut, diag_sum = best_cont_matrix(Y, y_estim)
    return permut

if __name__ == '__main__':
    dim_range = [2]
    N_range = [100, 1000]
    k_range = [2, 4]  #
    results = {}
    for n in N_range:
        for k in k_range:
            for d in dim_range:
                print "*******Computing for d =",d," k =",k," and ",n," points********"
                for _ in range(20):
                    main(d, k, n)
    dim_range = [5]
    N_range = [100, 1000, 5000]
    k_range = [4, 10, 20]  #
    results = {}
    for n in N_range:
        for k in k_range:
            for d in dim_range:
                print "*******Computing for d =",d," k =",k," and ",n," points********"
                for _ in range(20):
                    main(d, k, n)
    dim_range = [10]
    N_range = [100, 1000, 5000]
    k_range = [4, 10, 20, 50]  #
    results = {}
    for n in N_range:
        for k in k_range:
            for d in dim_range:
                print "*******Computing for d =",d," k =",k," and ",n," points********"
                for _ in range(100):
                    main(d, k, n)
    dim_range = [50]
    N_range = [100, 1000, 5000]
    k_range = [20, 50]  #
    results = {}
    for n in N_range:
        for k in k_range:
            for d in dim_range:
                print "*******Computing for d =",d," k =",k," and ",n," points********"
                for _ in range(100):
                    main(d, k, n)