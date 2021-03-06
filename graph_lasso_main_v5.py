from time import time

import numpy as np
from sklearn.mixture import GMM

from graph_lassov5 import GraphLassoMix
from tools.gm_tools_old import gaussian_mixture_sample, gm_params_generator, best_cont_matrix

def main(d,k,N):
    weights, centers, cov = gm_params_generator(d, k)

    X, Y = gaussian_mixture_sample(weights, centers, cov, N)
    lasso = GraphLassoMix(n_components=k, n_iter=20)
    t1_lasso = time()
    lasso.fit(X)
    t2_lasso=time()
    y_lasso = lasso.clusters_assigned

    gmm = GMM(n_components=k, covariance_type="full")
    t1_em= time()
    gmm.fit(X)
    y_em = gmm.predict(X)
    t2_em = time()

    # print "===Glasso result==="
    permut_lasso = algo_score(Y, y_lasso, t2_lasso - t1_lasso)

    # print "\n===EM result==="
    permut_gmm = algo_score(Y, y_em, t2_em - t1_em)
    # print weights
    # print lasso.pi[permut_lasso[0]]
    #print gmm.weights_[permut_gmm[0]]

    t = 0
    l = []
    for i in permut_lasso:
        l.append(1. / (cov[t].shape[0] ** 2) * np.linalg.norm(np.linalg.inv(cov[t]) - lasso.omegas[i]))
        t += 1
        #
    t2 = 0
    l2 = []
    for i in permut_gmm:
        print gmm.covars_[i].shape
        l2.append(1. / (cov[t2].shape[0] ** 2) * np.linalg.norm(np.linalg.inv(cov[t2]) - np.linalg.inv(gmm.covars_[i])))
        t2 += 1
    return max(l), max(l2)



def algo_score(Y, y_estim, t):
    mat, permut, diag_sum = best_cont_matrix(Y, y_estim)
    # print "best cont Matrix: "
    # print mat
    # print "Best Permutation: "
    # print permut
    # print "Diagonal Sum:"
    # print diag_sum
    # print "Correctly assigned ratio: "
    # print 1.0*diag_sum/len(Y)
    # print "Elapsed time:"
    # print t, 's'
    return permut

if __name__ == '__main__':
    dim_range = [4]
    N_range = [100]
    k_range = [2]  #
    results = {}
    for n in N_range:
        for k in k_range:
            for d in dim_range:
                print "*******Computing for d =",d," k =",k," and ",n," points********"
                lasso, gmm = main(d, k, n)
                results[n] = (lasso, gmm)
    print results
                #brute code to avoid errors
                #err = None
                #while err == None:
                #    try:
                #     #   main(d, k, n)
                #        err = 0
                #    except Exception as inst:
                #        template = "An exception of type {0} occured. Arguments:\n{1!r}"
                #        print template.format(type(inst).__name__, inst.args)
                #        pass

