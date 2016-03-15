from graph_lassov2 import lassoEM, gaussM, gaussM_Md
from pylab import *
from numpy import linalg as LA
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

n = 1000
lambd = [0.1, 1, 10 ,20]
weigths = [0.2, 0.8]
centers = [[-1, -1], [2,2]]
vars = [[[0.5, 0],[0, 0.5]], [[1,0],[0,0.5]]]


xy = 6*np.random.rand(n,2)-3
X_train = gaussM(weigths,centers,vars, n)
l = len(lambd)
z = gaussM_Md(xy, weigths, centers, vars)

fig = plt.figure()
plt.subplot(l+1, 2, 1)
plt.plot(xy[:,0], z, 'go')
plt.ylabel('Original x')
plt.subplot(l+1, 2, 2)
plt.ylabel('Original y')
plt.plot(xy[:,1], z, 'go')

z2 = []
for k in range(len(lambd)):
 lasso = lassoEM(n_components=2, n_iter=10, lambd=lambd[k])
 lasso.fit(X_train)

 print "Lasso =========" + str(lambd[k])
 print "\ncovars "
 for var in lasso.omega_:
     print "precision matrix"
     for i in var:
         print i

 print "\nEstimated weights :"
 print lasso.pi_
 print "\nEstimated variances :"
 for var in lasso.omega_:
     print LA.inv(np.array(var))
 print "\nEstimated means :"
 for mean in lasso.centers_:
     print mean
 #ax1 = fig.add_subplot(111, projection='3d')
 #ax1.scatter(xy[:,0],xy[:,1], z, c='r', marker='o')
 ## The triangles in parameter space determine which x, y, z points are
 ## connected by an edg
 #
 z2.append(gaussM_Md(xy, lasso.pi_, lasso.centers_, [LA.inv(np.array(om)) for om in lasso.omega_]))
 #ax2 = fig.add_subplot(222, projection='3d')
 #ax2.scatter(xy[:,0],xy[:,1], z2, c='b', marker='o')
 #plt.show()
 #x,y = np.meshgrid(xy[:,0],xy[:,1])
 #
 #ax = fig.gca(projection='3d')
 #surf = ax.plot_surface(x,y, z, rstride=1, cstride=1, cmap=cm.coolwarm,
 #        linewidth=0, antialiased=False)
 #ax.set_zlim(0, 1.0)
 #
 #fig.colorbar(surf, shrink=0.5, aspect=5)

 plt.subplot(l+1, 2, 2*(k+1)+1)
 plt.plot(xy[:,0], z2[k], 'ro')
 plt.ylabel(str(lambd[k]))
 plt.subplot(l+1, 2, 2*(k+1)+2)
 plt.plot(xy[:,1], z2[k], 'ro')
plt.show()
