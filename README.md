# Clustering models
This repository contains code written during my Ph.D thesis on clustering models for Gaussian mixture models.

Note: this code is provided 'as-is' and need refactoring to be used. It relies heavily on Numba, comment @jit decorator for testing. 

Several portions of this code come from different sources, some of which I did not referenced properly, nevertheless, I thank the authors.

the models present in this repository are:

- Graphical lasso on Gaussian Mixture models
- Square root lasso on Gaussian Mixture models
- Estimation of number of clusters by regularization of posterior probabilities
- Estimation of number of clusters by penalizing the weight vector.

I also implemented
- Projection on the probability simplex from http://arxiv.org/pdf/1309.1541.pdf in slope_implementation.py