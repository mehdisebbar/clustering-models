from gm_tools import gaussian_mixture_sample, gm_params_generator
from graph_lassov5 import GraphLassoMix
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pylab import *
from gm_tools import get_centers_order

d = 10
k=2
N= 10000
weights, centers, cov = gm_params_generator(d,k)

X = gaussian_mixture_sample(weights, centers, cov, N)
lasso = GraphLassoMix(n_components=k, n_iter=5)
lasso.fit(X)

clusters_match_list = get_centers_order(centers, lasso.centers)

for i, (u,v) in enumerate(clusters_match_list):
    print "----Cluster ",i
    print "Real Center: ",centers[u]
    print "Estimated Center:",lasso.centers[v]

matched_matrix_diff = [np.abs(np.linalg.inv(cov[u])-lasso.omegas[v]) for u,v in clusters_match_list]

fig = plt.figure()
for i,omega in enumerate(matched_matrix_diff):
    ax = fig.add_subplot(1,k+1,i+1)
    cax = ax.matshow(omega, interpolation='nearest')
fig.colorbar(cax)
plt.show()