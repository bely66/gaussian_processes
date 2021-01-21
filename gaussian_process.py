import matplotlib.pyplot as plt 
import numpy as np 
from numpy.linalg import inv
import random

# it's used as a covariance matrix 

'''
Radial Basis Function (RBF)
The maximum value that the RBF kernel can be is 1 and occurs when d₁₂ is 0 which is when the points are the same, i.e. X₁ = X₂.
1. When the points are the same, there is no distance between them and therefore they are extremely similar
2. When the points are separated by a large distance, 
   then the kernel value is less than 1 and close to 0 which would mean that the points are dissimilar.

For choosing more advanced kernels:
https://www.cs.toronto.edu/~duvenaud/cookbook/
'''
def rbf_kernel(x_1, x_2, l=1.0, sigma_f=1.0):
    # (x_1 - X_2)^2 = x_1^2 - 2*x_1*x_2 + x_2^2
    sqdist = np.sum(x_1**2, 1).reshape(-1, 1) - 2*np.dot(x_1, x_2.T) + np.sum(x_2**2, 1)
    return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)


def posterior(x_test, x_train, y_train, l=1.0, sigma_f=1.0, noise=0.06):
    k_train = rbf_kernel(x_train, x_train, l, sigma_f) + noise**2 * np.eye(len(x_train))
    k_joint = rbf_kernel(x_train, x_test, l, sigma_f)
    k_test = rbf_kernel(x_test, x_test, l, sigma_f) + 1e-8 * np.eye(len(x_test))
    k_train_inv = inv(k_train)

    mu_joint = k_joint.T.dot(k_train_inv).dot(y_train)
    cov_joint = k_test - k_joint.T.dot(k_train_inv).dot(k_joint)

    return mu_joint, cov_joint

plt.figure()
#no of bins
dims = 30

x_train = np.linspace(1, 10, dims).reshape(-1, 1)
y_train = np.sin(x_train) + np.sin(20 * x_train) + np.cos(300 * x_train) + 2
x_test = np.linspace(1, 10, 200).reshape(-1, 1)
plt.plot(x_train, y_train)
plt.savefig("data_plot.png")

plt.figure()
# covariance 
cov = rbf_kernel(x_train, x_train, l=0.5, sigma_f=0.5)

gp1 = np.dot(cov, y_train.reshape(-1))
plt.scatter(x_train, y_train)
for i in range(5):
    samples = np.random.multivariate_normal(mean=gp1,cov=cov)
    plt.plot(x_train, samples)
plt.savefig("untrained_gaussians.png")

'''
mu_[2|1]  =  0 + K_[1,2] * inv(K_[1,1]) * (y1 - 0)
          =      K_[1,2] * inv(K_[1,1]) *  y1
cov_[2|1] = K_[2,2] - K_[1,2] * inv(K_[1,1]) * K_[1,2]
'''



plt.figure()

mu_test, cov_test = posterior(x_test,x_train,y_train)
samples = np.random.multivariate_normal(mu_test.transpose()[0],  
                                        cov_test, 5)
plt.scatter(x_train,y_train)
for i in range(len(samples)):
    plt.plot(x_test,samples[i])

plt.savefig("trained_gaussian.png")
