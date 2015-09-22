from graph_lasso import graphlassoEM, gaussM
from numpy.linalg import inv
import numpy as np
from sklearn.mixture import GMM

if __name__ == '__main__':
    mu1=[1,2,6]
    mu2=[5,6,1]
    mu3=[4,5,2]
    sigma1= [[10,1,2],[1,2,4],[2,4,9]]
    sigma2= [[3,2,5],[2,6,4],[5,4,10]]
    sigma3= [[9,1,1],[1,4,1],[1,1,5]]
    mu=[mu1,mu2, mu3]
    sigmas=[sigma1, sigma2, sigma3]
    X =gaussM([0.2, 0.3, 0.5], mu, sigmas, 200)
    om, muhat, pi = graphlassoEM(X, 3, [20,20, 10])
    print "real omega, \n", sigma1
    print "real omega, \n", sigma2
    print "real omega 3: \n", sigma3
    print "----------------------------"
    print "estimated omega, \n", om[1]
    print "estimated omega, \n", om[0]
    print 'estimated omega3: \n',om[2]
    print "----------------------------"
    print "real mu, \n", mu1
    print "real mu, \n", mu2
    print 'real mu:\n', mu3
    print "----------------------------"
    print "estimated mu3: \n", muhat[2]
    print "estimated mu1, \n",muhat[0]
    print "estimated mu1, \n",muhat[1]
    print "----------------------------"
    print pi
