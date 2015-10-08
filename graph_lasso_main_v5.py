from gm_tools import gaussian_mixture_sample, gm_params_generator, match_labels, clusters_stats, cluster_match
from graph_lassov5 import GraphLassoMix
import numpy as np
from gm_tools import get_centers_order
import matplotlib.pyplot as plt

def main(d,k,N):

    weights, centers, cov = gm_params_generator(d,k)

    X, Y = gaussian_mixture_sample(weights, centers, cov, N)
    lasso = GraphLassoMix(n_components=k+1, n_iter=5)
    lasso.fit(X)
    y = lasso.clusters_assigned
    print set(y)
    clusters_match_list = get_centers_order(centers, lasso.centers)
    y_matched = match_labels(y, clusters_match_list)
    u,v = cluster_match(Y,y)
    print u, v
    return clusters_stats(Y, y_matched)

score_list = []
dim_range = [2, 5,10,20, 40]
for d in dim_range:
    score_list.append(main(d, 4, 100000))
plt.plot(dim_range, score_list, 'r--')
plt.axis([min(dim_range)-10, max(dim_range)+10, 0, 1.2])
plt.show()



#clusters_match_list = get_centers_order(centers, lasso.centers)
#
#for i, (u,v) in enumerate(clusters_match_list):
#    print "----Cluster ",i
#    print "Real Center: ",centers[u]
#    print "Estimated Center:",lasso.centers[v]
#
#matched_matrix_diff = [np.abs(np.linalg.inv(cov[u])-lasso.omegas[v]) for u,v in clusters_match_list]
#
#fig = plt.figure()
#for i,omega in enumerate(matched_matrix_diff):
#    ax = fig.add_subplot(1,k+1,i+1)
#    cax = ax.matshow(omega, interpolation='nearest')
#fig.colorbar(cax)
#plt.show()