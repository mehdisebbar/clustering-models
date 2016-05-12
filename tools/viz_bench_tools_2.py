import numpy as np
from sklearn import datasets
from sklearn.mixture import GMM
from scipy.stats import multivariate_normal
from sklearn.cross_validation import train_test_split
from gm_tools import gaussian_mixture_sample, gm_params_generator, gauss_mix_density

class Benchmark():
    """
    A benchmarking tool to evaluate the performance of a clustering algorithm on gaussian mixtures
    with a regularization parameter compared to EM on different datasets.
    
    1- Generate dict of dataset with format : 
    {"dataset_name": {"data": data_array, "target": target_array, "params":{} }}}
    2- Loop on each dataset and evaluate the algorithm and EM
        a- for an algorithm without the param k and a regularization param lambda_param select the best lambda by 
            - splitting the dataset in two sets, estimate the params with the algorithm on the first dataset
              for different values of lambda_param
            - select the lambda_param that maximize the likelihood on the 2nd dataset
            - return this lambda_param and related parameters estimators.
    3- Generate a visualization of the results
    """
    def __init__(self, 
                 algo, 
                 gaussian_mix_K = [2, 4], 
                 gaussian_mix_dim = 4, 
                 gaussian_mix_N = 1000, 
                 lambda_param_list = [1],
                 max_clusters =10,
                 n_algo_iter = 10
                ):
        self.gaussian_mix_K = gaussian_mix_K
        self.gaussian_mix_dim = gaussian_mix_dim
        self.gaussian_mix_N = gaussian_mix_N
        self.lambda_param_list = lambda_param_list
        self.max_clusters = max_clusters
        self.n_algo_iter = n_algo_iter
        
    def run(self):
        pass
    
    def dataset_gen(self):
        """
        Dataset generation of :
            - Gaussian mixture with different values of k
            - Iris dataset from sklearn        
        """

        dataset_list = []
        #Generating a gaussian mixture sample
        def gaussian_mix_gen(k, d, N):
            pi_, means_, covars_ = gm_params_generator(d, k)
            X,y = gaussian_mixture_sample(pi_, means_, covars_, N)
            return {"gaussian_mix_"+ str(k): {
                    "data": X,
                    "target": y,
                    "params": {
                        "covars": covars_,
                        "centers": means_,
                        "weights": pi_,
                        "clusters": k}}}
     
        dataset_list += [gaussian_mix_gen(k, self.gaussian_mix_dim, self.gaussian_mix_N) 
                         for k in self.gaussian_mix_K]
    
        #Iris dataset 
        dataset_list.append({"iris": {
                    "data": datasets.load_iris().data, 
                    "target": datasets.load_iris().target}
                            }
                           )
            
        #final dataset
        final_dataset = {}
        for dst in dataset_list:
            final_dataset = merge_two_dicts(final_dataset, dst)
        return final_dataset
    
    def algo_eval(self, algo, X):
        """
        Evaluate the algorithm with EM on a specific dataset X
        returns parameters estimators
        """
        # chose the best lambda_param according to the lambda_param_select function 
        # and fit the algorithm
        best_lambda_param = self.lambda_param_select(X, algo, self.lambda_param_list)
        alg = algo(lambda_param = best_lambda_param, n_iter=self.n_algo_iter, max_clusters= self.max_clusters)
        weights_algo, _, centers_algo, covars_algo = alg.fit(X)
        #We evaluate EM with the number of clusters given by the previous algorithm
        em = GMM(n_components=len(weights_algo), covariance_type="full")
        em.fit(X)
        res =  {
            "algo": {
                "best":{
                    "weights" : weights_algo,
                    "centers" : centers_algo,
                    "covars" : covars_algo,
                    "lambda": best_lambda_param
                },
                "others":{
                    
                }
            },
            "em": {
                "weights" : em.weights_,
                "centers" : em.means_,
                "covars" : em.covars_,
            }
        }
        return res

    
    def lambda_param_select(self, X, algo, lambda_param_list):
        """
        We split the dataset onto 2 sets, we will estimate parameters of the gaussian_mix for 
        differents values of lambda_param on the first dataset and chose the lambda_param that
        maximize the log-likelihood on the 2nd set
        
        the algorithm needs a lambda_param in argument and must return (weights, _, centers, covars)
        
        returns a lambda_param for the algorithm
        """
        X1, X2, _,_ =train_test_split(X,np.zeros(len(X)), test_size = 0.25)
        #We will store results in a dict, {log-likelihood:lambda_param}
        results = {}
        for lambd_param in lambda_param_list:
            #fit the algorithm on the first dataset X1
            alg = algo(lambda_param = lambd_param, n_iter=self.n_algo_iter, max_clusters=self.max_clusters)
            weights, _, centers, covars = alg.fit(X1)
            print "lambda:", weights
            print "centers:", centers
            #evaluate the log-likelihood on the second dataset X2
            results[self.gm_loglikelihood(X2, weights, centers, covars)] = lambd_param
        return results[max(results.keys())]
    
    def gm_loglikelihood(self, X, weights, centers, covars):
        k = len(weights)
        return np.array([
            np.log(np.array([weights[j] * multivariate_normal.pdf(x, centers[j], covars[j]) for j in range(k)]).sum())
            for x in X
        ]).sum()
          
    def visualize_scores(self):
        pass
        
    def merge_two_dicts(self, x, y):
        '''Given two dicts, merge them into a new dict as a shallow copy.'''
        z = x.copy()
        z.update(y)
        return z
    def score_eval(self,res):
        pass
        
