import sys
algo_root = '..'
sys.path.insert(0, algo_root)
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from tools.gm_tools import gm_params_generator, gaussian_mixture_sample
from tools.gm_tools import score
from cluster.sq_root_lasso import sqrt_lasso_gmm
from sklearn.mixture import GMM
import pickle

data_size_list = [5 * 1e3, 5 * 1e4, 5 * 1e5]
cluster_size_list = [3, 5, 8]
dim_list = [2, 4, 6]
max_cluster_increment = 4

def getkey(item):
    return item[0]

def weights_compare(pi1, pi2):
    if len(pi1) == len(pi2):
        return ((np.array(pi1)-np.array(pi2))**2).sum()
    elif len(pi1) < len(pi2):
        return ((np.array(pi1+[0]*(len(pi2)-len(pi1)))-np.array(pi2))**2).sum()
    else:
        return ((np.array(pi2+[0]*(len(pi1)-len(pi2)))-np.array(pi1))**2).sum()

#we define a bic scoring method for the grid search
def bic_scorer(estimator, X, y=None):
    try:
        return (2 * score(X, estimator.weights_, estimator.means_, estimator.covars_) -
            estimator._n_parameters()*np.log(X.shape[0]))
    except:
        print "Unexpected scoring error:", sys.exc_info()[0]
        return -9 * 1e5


for data_size in data_size_list:
    for cluster_size in cluster_size_list:
        for dim in dim_list:
            try:
                pi, means, covars = gm_params_generator(dim, cluster_size, min_center_dist=0.1)
                # pi = np.array([0.2 , 0.4, 0.4])
                # means = np.array( [[0,0],[2,2],[2,0]])
                X, _ = gaussian_mixture_sample(pi, means, covars, data_size)
                # view2Ddata(X)

                test_size = 0.2
                X_train, X_validation, y_train, y_test = train_test_split(
                    X, np.zeros(len(X)), test_size=test_size, random_state=0)

                # grid search on sq_root_lasso method
                max_clusters = cluster_size + max_cluster_increment
                lambd = np.sqrt(2 * np.log(max_clusters) / X_train.shape[0])
                param = {"lambd": [lambd * 1e-1, lambd, lambd * 1e1], "lipz_c": [1, 1e-1, 5],
                         "max_clusters": [max_clusters]}
                clf = GridSearchCV(estimator=sqrt_lasso_gmm(n_iter=100), param_grid=param, cv=3, n_jobs=-2,
                                   scoring=bic_scorer)
                clf.fit(X_train, y_train)

                params_GMM = {"n_components": range(2, max_clusters + 1)}
                clf_gmm = GridSearchCV(GMM(covariance_type='full'), param_grid=params_GMM, cv=3, n_jobs=-2,
                                       scoring=bic_scorer)
                clf_gmm.fit(X_train)

                # we evaluate the loglikelihood of the fitted models on X_validation
                print "X_validation/X_train ratio: ", test_size

                liklhd_r = 1. / X_validation.shape[0] * score(X_validation, pi, means, covars)
                print "real loglikelihood: ", liklhd_r
                weights_s, means_s = map(list, zip(*(sorted(zip(pi, means), key=getkey)[::-1])))
                print "real pi:", np.array(weights_s)
                print "real means:", np.array(means_s), "\n"
                print "### sq_root lasso method ###\n"
                print "crossval + gridsearch params: ", clf, "\n"
                print "grid search best params:", clf.best_params_, "\n"

                weights_sqrt, means_sqrt, covars_sqrt = map(list, zip(*(
                sorted(zip(clf.best_estimator_.weights_, clf.best_estimator_.means_, clf.best_estimator_.covars_),
                       key=getkey)[::-1])))
                liklhd_sqrt = 1. / X_validation.shape[0] * score(X_validation, clf.best_estimator_.weights_,
                                                                 clf.best_estimator_.means_,
                                                                 clf.best_estimator_.covars_)
                print "sq_root lasso method loglikelihood on X_validation:", liklhd_sqrt
                print "likelihood_diff: ", liklhd_r - liklhd_sqrt, "\n"

                print "sorted pi: ", np.array(weights_sqrt)
                print "sorted means: ", np.array(means_sqrt), "\n"
                print "norm(weights_sqrt-weights_real) :", weights_compare(weights_sqrt, weights_s), "\n"

                print "### EM + BIC ###\n"
                print "crossval + gridsearch params: ", clf_gmm, "\n"
                print "grid search best params:", clf_gmm.best_params_, "\n"

                weights_gmm, means_gmm, covars_gmm = map(list, zip(*(sorted(
                    zip(clf_gmm.best_estimator_.weights_, clf_gmm.best_estimator_.means_,
                        clf_gmm.best_estimator_.covars_), key=getkey)[::-1])))
                liklhd_gmm = 1. / X_validation.shape[0] * score(X_validation, clf_gmm.best_estimator_.weights_,
                                                                clf_gmm.best_estimator_.means_,
                                                                clf_gmm.best_estimator_.covars_)
                print "EM loglikelihood on X_validation:", liklhd_gmm
                print "likelihood_diff: ", liklhd_r - liklhd_gmm, "\n"

                print "sorted pi: ", np.array(weights_gmm)
                print "sorted means: ", np.array(means_gmm), "\n"
                print "norm(weights_gmm-weights_real) :", weights_compare(weights_gmm, weights_s), "\n"
                pickle.dump({"real_cluster_size": cluster_size,
                             "size": data_size,
                             "real":
                                 {
                                     "X": X,
                                     "weights": pi,
                                     "means": means,
                                     "covars": covars
                                 },
                             "sqrt":
                                 {
                                     "clf": clf,
                                     "score_weights": weights_compare(weights_sqrt, weights_s),
                                     "weights": weights_sqrt,
                                     "means": means_sqrt,
                                     "covars": covars_sqrt
                                 },
                             "gmm":
                                 {
                                     "gmm_clf": clf_gmm,
                                     "score_weights": weights_compare(weights_gmm, weights_s),
                                     "weights": weights_gmm,
                                     "means": means_gmm,
                                     "covars": covars_gmm
                                 }

                             }, open("res_" + "D" + str(dim) + "K" + str(cluster_size) + "N" + str(data_size), "wb"))
                print "Done for ", "res_" + "D" + str(dim) + "K" + str(cluster_size) + "N" + str(data_size)
            except:
                print "error on calc of: ", " data_size: ", str(data_size), " cluster_size: ", str(
                    cluster_size), " dim: ", str(dim)
                print "Unexpected error:", sys.exc_info()[0]
                pass
