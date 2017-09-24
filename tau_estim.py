import numpy as np
from cvxpy import *
from numba import jit
from scipy.stats import multivariate_normal

class EMtau(EM):
    # A EM-like algorithm that penalizes the columns of the posterior matrix tau
    # resulting to a "sparse" estimation of the weights i.e. a estimation of the number of clusters
    
    def __init__(self, kmax=2, n_iter=10):
        super(EMtau, self).__init__(kmax, n_iter)
        self.fista_iter = 20
        self.lambd = 1
    
    @jit
    def f_gradient(self, xi, X):
        #give gradient of f on xi
        temp = np.zeros([self.kmax, self.N])
        for i in range(self.kmax):
            temp[i] = -np.log(multivariate_normal(self.centers[i], self.covars[i]).pdf(X)) - np.log(self.pi[i]/xi[:,i]) +1 + xi[:,i]/(np.linalg.norm(xi, axis=0)[i])
        return temp.T
    
    def expectation(self, X):
        # Estimation of tau with penalization on columns
        # We use CVXPY, a more efficient implementation would be to solve directly the optimization problem
        # with the appropriate procedure
        t_current = 1
        xi = super(EMtau, self).tau_gen(X)
        tau_current = np.copy(xi)
        tau_next = np.ones([self.N, self.kmax])
        i=0
        # The thresholds given are purely experimental
        while np.linalg.norm(tau_current-tau_next) > 1e-5 and i < self.fista_iter:
            print np.linalg.norm(tau_current-tau_next)
            print "iter", i
            tau_current = np.copy(tau_next)
            tau = [[Variable() for _ in range(self.kmax) ] for _ in range(self.N)]
            constraints = [sum_entries(bmat(ligne_n)) == 1 for ligne_n in tau ]+[item >=0 for sublist in tau for item in sublist ]             
            f_grad = self.f_gradient(xi, X)
            #import pdb; pdb.set_trace()
            xi_next = xi-self.lambd*f_grad
            prob = Problem(Minimize(norm(bmat(tau) - xi_next )**2), constraints)
            prob.solve(solver=SCS, use_indirect=True)
            tau_next = np.array(bmat(tau).value)+1e-20
            t_next = (1+np.sqrt(1+4*t_current*2))/2
            xi = tau_next + (t_current - 1)/t_next*(tau_next - tau_current)
            # We set the negative values to 1e-20 to avoid errors with log in the gradient
            xi[xi <= 0] = 1e-20
            t_current = t_next
            i+=1
        return tau_next
        
    