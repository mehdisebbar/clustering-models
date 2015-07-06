from graph_lasso import graphlassoEM, gaussM
from numpy.linalg import inv

if __name__ == '__main__':
    mu1=[1,2]
    mu2=[5,6]
    sigma1= [[3,1],[1,3]]
    sigma2= [[3,2],[2,6]]
    mu=[mu1,mu2]
    sigmas=[sigma1, sigma2]
    X =gaussM([0.4, 0.6], mu, sigmas, 10)
    om, muhat, pi = graphlassoEM(X, 2, [0.01,0.01])
    print "real omega, ", inv(sigma1)
    print "estimated omega, ", om[0]
    print "real omega, ", inv(sigma2)
    print "estimated omega, ", om[1]
    print "real mu, ", mu1
    print "estimated mu1, ",muhat[0]
    print "real mu, ", mu2
    print "estimated mu1, ",muhat[1]
    print pi
