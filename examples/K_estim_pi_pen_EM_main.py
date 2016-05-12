import numpy as np

from K_estim_pi_pen_EM import GraphLassoMix
from tools.gm_tools import gaussian_mixture_sample, gm_params_generator, gauss_mix_density

d = 5
k = 3
N = 10000
pi, means, covars = gm_params_generator(d, k)
print "original weights:", pi
X, Y = gaussian_mixture_sample(pi, means, covars, N)
g = GraphLassoMix(n_iter=10)
pi_estim, _, means_estim, covars_estim = g.fit(X)

print "vraies valeurs: ", pi

print "erreur:"
print np.array([(gauss_mix_density(x, pi_estim, means_estim, covars_estim) - gauss_mix_density(x, pi, means, covars)) ** 2 for x in X]).sum()