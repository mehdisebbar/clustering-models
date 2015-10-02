from gm_tools import gaussian_mixture_sample, gm_params_generator
from graph_lassov5 import GraphLassoMix
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pylab import *

d = 20
k=4
N= 100000
weights, centers, cov = gm_params_generator(d,k)
print centers

X = gaussian_mixture_sample(weights, centers, cov, N)
lasso = GraphLassoMix(n_components=k, n_iter=10)
lasso.fit(X)
res = np.abs(lasso.omegas-np.array([np.linalg.inv(c) for c in cov]))

print "Real centers"
for center in centers:
    print center

print "Estimated centers"
for center in lasso.centers:
    print center

fig = plt.figure()
for i,omega in enumerate(res):
    ax = fig.add_subplot(1,k,i)
    cax = ax.matshow(omega, interpolation='nearest')
    fig.colorbar(cax)
plt.show()