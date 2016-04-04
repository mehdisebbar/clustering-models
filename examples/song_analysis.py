import pandas as pd
import numpy as np
dfrh = pd.read_csv("./songs/feats.rh",header=None )
dfrp = pd.read_csv("./songs/feats.rp",header=None )
dfssd = pd.read_csv("./songs/feats.ssd",header=None )
dfssd.as_matrix()[:,1:].shape
X = np.hstack([np.hstack([dfrh.as_matrix()[:,1:], dfrp.as_matrix()[:,1:]]),dfssd.as_matrix()[:,1:]])
from sklearn.decomposition import PCA
p = PCA(n_components=10)
X2 = p.fit_transform(X)
from scipy.spatial.distance import pdist, squareform
pdist(X2, 'euclidean')
from K_estim_pi_pen_EM import GraphLassoMix

alg = GraphLassoMix(lambda_param=0, n_iter=20, max_clusters=5)
res = alg.fit(X2)