from K_estim_pi_pen_EM import GraphLassoMix
from gm_tools import gaussian_mixture_sample, gm_params_generator

d = 5
k = 3
N = 10000
weights, centers, cov = gm_params_generator(d,k)
print "original weights:", weights
X, Y = gaussian_mixture_sample(weights, centers, cov, N)
g = GraphLassoMix(n_iter=10)
a = g.fit(X)