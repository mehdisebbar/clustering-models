# coding: utf-8

# In[47]:

import cvxpy as cvx
import numpy as np
import scipy as scipy

"""
# Fix random number generator so we can repeat the experiment.
np.random.seed(0)

# Dimension of matrix.
n = 10

# Number of samples, y_i
N = 10000

# Create sparse, symmetric PSD matrix S
A = np.mat(np.random.randn(n, n))  # Unit normal gaussian distribution.
A[scipy.sparse.rand(n, n, 0.85).todense().nonzero()] = 0  # Sparsen the matrix.
Strue = A*A.T + 0.05 * np.matrix(np.eye(n))  # Force strict pos. def.

# Create the covariance matrix associated with S.
R = np.linalg.inv(Strue)

# Create samples y_i from the distribution with covariance R. 
y_sample = scipy.linalg.sqrtm(R) * np.matrix(np.random.randn(n, N))

# Calculate the sample covariance matrix.
Y = np.cov(y_sample)


# In[48]:

A*A.T


# In[49]:

A*A.T + 0.05 * np.matrix(np.eye(n))


# In[50]:

# The alpha values for each attempt at generating a sparse inverse cov. matrix.
alphas = [10, 2, 1]
vlambda = [0, 1, 10]

# Empty list of result matrixes S
Ss = {}

# Solve the optimization problem for each value of alpha.
for alpha in alphas:
    for beta in vlambda:
    # Create a variable that is constrained to the positive semidefinite cone.
        S = cvx.semidefinite(n)
    
    # Form the logdet(S) - tr(SY) objective. Note the use of a set
    # comprehension to form a set of the diagonal elements of S*Y, and the
    # native sum function, which is compatible with cvxpy, to compute the trace.
    # TODO: If a cvxpy trace operator becomes available, use it!
        obj = cvx.Minimize(-cvx.log_det(S) - sum([(S*Y)[i, i] for i in range(n)]) + beta * cvx.norm(S, 1))
    
    # Set constraint.
        constraints = [cvx.sum_entries(cvx.abs(S)) <= alpha]
    
    # Form and solve optimization problem
        proba = cvx.Problem(obj)
        prob.solve()
        if prob.status != cvx.OPTIMAL:
            print "Noooooo"
            raise Exception('CVXPY Error')
        print S.value

    # If the covariance matrix R is desired, here is how it to create it.
        #R_hat = np.linalg.inv(S.value)
    
    # Threshold S element values to enforce exact zeros:
        #S = S.value
        #S[abs(S) <= 1e-4] = 0

    # Store this S in the list of results for later plotting.
        Ss[(alpha, beta)] = S.value
        

        print 'Completed optimization parameterized by alpha =', alpha, ' and lambda: ', beta


# In[51]:

import matplotlib.pyplot as plt

# Show plot inline in ipython.

# Plot properties.
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Create figure.
plt.figure()
plt.figure(figsize=(12, 12))

# Plot sparsity pattern for the true covariance matrix.
plt.subplot(2, 2, 1)
plt.spy(Strue)
plt.title('Inverse of true covariance matrix', fontsize=16)

# Plot sparsity pattern for each result, corresponding to a specific alpha.
for i in range(len(alphas)):
    plt.subplot(2, 2, 2+i)
    plt.spy(Ss[i])
    plt.title('Estimated inv. cov matrix, $\\alpha$={}'.format(alphas[i]), fontsize=16)

"""
# In[52]:

def graphlasso(X, tauk, muk, lambd):  # tauk nxk matrix, muk: pxk
    Ss = []
    for k in range(len(muk)):
        Xc = X - muk[k]
        cluster_cov = np.array([np.array(np.mat(vect).T*np.mat(vect)) for vect in Xc]) #check perf with np.dot(X.T, X) and take diag
        Snk = np.dot(cluster_cov.T, tauk[:, k])
        S = cvx.semidefinite(len(muk[0]))
        obj = cvx.Maximize(cvx.log_det(S) - sum([(Snk*S)[i, i] for i in range(len(muk))]))

        # Set constraint.
        constraints = [cvx.sum_entries(cvx.abs(S)) <= lambd[k]]

        # Form and solve optimization problem
        prob = cvx.Problem(obj, constraints)
        prob.solve()
        if prob.status != cvx.OPTIMAL:
            raise Exception('CVXPY Error')

            # If the covariance matrix R is desired, here is how it to create it.
            # R_hat = np.linalg.inv(S.value)

            # Threshold S element values to enforce exact zeros:
        S = S.value
        S[abs(S) <= 1e-4] = 0

        # Store this S in the list of results for later plotting.
        Ss += [S]
    return Ss


"""
# In[60]:


a=np.random.random_sample(4)
print a


# In[221]:
"""


def gauss_dens_inv(mu, omega, x):
    return np.sqrt(np.linalg.det(omega) / ((2 * np.pi) ** len(x))) * np.exp(
        -0.5 * np.dot(np.dot((x.T - mu), omega), (x.T - mu).T))

# In[73]:
"""
pi = np.random.multinomial(1000, [0.2, 0.4, 0.1 ,0.3], size =1)


# In[101]:
"""


def gaussM(pi, mu, sigmas, n):
    pi = np.random.multinomial(n, pi, size=1)
    S = np.random.multivariate_normal(mu[0], sigmas[0], pi[0][0])
    for idx, nk in enumerate(pi[0][1:]):
        S = np.vstack([S, np.random.multivariate_normal(mu[idx], sigmas[idx], nk)])
    return S

# In[102]:
"""
np.random.multivariate_normal([1,2],[[3,2],[2,3]],2).T


# In[103]:

for idx, val in enumerate([1,2]):
    print idx, val


# In[104]:

mu1=[1,2]
mu2=[5,6]
sigma1= [[3,1],[1,3]]
sigma2= [[3,2],[2,6]]
mu=[mu1,mu2]
sigmas=[sigma1, sigma2]


# In[112]:

X =gaussM([0.4, 0.6], mu, sigmas, 100)


# In[119]:
"""


def omega_sample(p):
    A = np.mat(np.random.randn(p, p))  # Unit normal gaussian distribution.
    # A[scipy.sparse.rand(p, p, 0.85).todense().nonzero()] = 0  # Sparsen the matrix.
    return A * A.T


# In[228]:

def graphlassoEM(X, K, lambd):
    dim = len(X[0])  # dimension
    tau = np.random.rand(len(X), K)  # init tau_i_k
    a = np.random.random_sample(K)  # init pi
    pi = a / sum(a)  # pi
    mu = np.random.random_sample((K, dim))  # K centers of R^dim, BETWEEN  0-1
    omega = [omega_sample(dim) for _ in range(K)]
    for _ in range(100):  # iterations EM procedure
        for i in range(len(X)):  # estimation tau_i_k
            for k in range(K):  # for each cluster
                b = 0
                for k in range(K):
                    b += pi[k] * gauss_dens_inv(mu[k], omega[k], X[i])
                tau[i][k] = pi[k] * gauss_dens_inv(mu[k], omega[k], X[i]) / b
        pi = 1.0 / len(X) * np.sum(tau, axis=0)
        print pi
        for k in range(K): #estimates mu
            a = tau[:, k] #get column tau on k cluster
            c = (a * X.T).T #get tau_i_k * X
            mu[k] = 1.0 / (len(X) * pi[k]) * np.sum(c, axis=0)
        omega = graphlasso(X, tau, mu, lambd)
    return omega, mu, pi

# In[229]:
"""
a=np.random.random_sample(2)
c=    pi_0=a/sum(a)
print pi_0


# In[230]:

graphlassoEM(X, 2, [1,1])


# In[154]:

np.random.random_sample((2, 3))


# In[149]:

c=[[ -2.31584178e+077,  -2.00389857e+000],
 [  2.19649460e-314 ,  2.14905802e-314],
 [  0.00000000e+000  , 0.00000000e+000],
 [              3   ,0.00000000e+000],
 [  0.00000000e+000  , 4.44659081e-323],
 [  2.19649585e-314   ,6.93967973e-310],
 [  0.00000000e+000   ,0.00000000e+000]]


# In[196]:

x=np.array([3,5])
mu=np.array([1,2])
omega=np.array([[3,2],[2,3]])


# In[197]:

np.exp(-0.5*np.dot(np.dot((x-mu),omega), (x-mu)))


# In[198]:

gauss_dens_inv(mu, omega, x)


# In[199]:

(np.linalg.det(omega)/((2*np.pi)**len(x)))*np.exp(-0.5*np.dot(np.dot((x-mu), omega), (x-mu).T))


# In[ ]:

"""
