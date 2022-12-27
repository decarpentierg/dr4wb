import numpy as np
import ot

np.random.seed(0)

n = 10
poly_n = 100
d = 300
m = 30

# Distributions of which to compute the barycenter
mu1 = np.random.normal(size=(poly_n, d))
mu2 = np.random.normal(size=(poly_n, d))
weights = np.ones(poly_n)

# Random projection matrix
A = np.random.normal(size=(m, d)) / np.sqrt(m)

# Projected distributions
mu1_ = mu1 @ A.T
mu2_ = mu2 @ A.T

# Distance matrices
M = ot.dist(mu1, mu2)
M_ = ot.dist(mu1_, mu2_)

# Optimal transport
gamma, log = ot.emd(weights, weights, M, log=True)
gamma_, log_ = ot.emd(weights, weights, M_, log=True)
