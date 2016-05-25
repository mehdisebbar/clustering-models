import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV

from cluster.sq_root_lasso import sqrt_lasso_gmm
from tools.algorithms_benchmark import view2Ddata
from tools.gm_tools import gm_params_generator, gaussian_mixture_sample

pi, means, covars = gm_params_generator(2, 3)
X, _ = gaussian_mixture_sample(pi, means, covars, 1e4)
view2Ddata(X)
X_train, X_validation, y_train, y_test = train_test_split(
    X, np.zeros(len(X)), test_size=0.2, random_state=0)
max_clusters = 8
lambd = np.sqrt(2 * np.log(max_clusters) / X_train.shape[0])
param = {"lambda_param": [0, 0.01, 0.1, 1], "Lipshitz_c": [1, 10, 100, 1000]}
clf = GridSearchCV(estimator=sqrt_lasso_gmm(lambda_param=1, Lipshitz_c=1, n_iter=100, max_clusters=8, verbose=False),
                   param_grid=param, cv=5, n_jobs=-1)
clf.fit(X_train, y_train)
print clf.best_params_
