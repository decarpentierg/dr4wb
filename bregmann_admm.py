import numpy as np

def centroid_sphBregman(stride, supp, w, c0, options):
    d = supp.shape[0]
    n = len(stride)
    m = len(w)
    posvec=[1,np.cumsum(stride)+1]

    if c0 == []:
        c = centroid_init(stride, supp, w, options)
    else:
        c = c0
    support_size = len(c.w)  

    X = np.zeros((support_size, m))
    Y = np.zeros_like(X)
    Z = np.copy(X)
    spIDX_rows = np.zeros((support_size * m, 1))
    spIDX_cols = np.zeros_like(spIDX_rows)
    for i in range(n):
        xx, yy = np.meshgrid((i-1)*support_size + np.arange(support_size), posvec[i]:posvec[i+1]-1)
        ii = support_size*(posvec[i]-1) + np.arange(support_size*stride[i])
        spIDX_rows[ii] = xx.T
        spIDX_cols[ii] = yy.T
    spIDX = np.kron(np.eye(support_size), np.ones((1, n)))
    
    # initialization
    for i in range(n):
        Z[:,posvec[i]:posvec[i+1]-1] = 1/(support_size*stride[i])
    C = np.linalg.norm(c.supp.T - supp.T, axis=0)**2

    nIter = 2000
    if "badmm_max_iters" in options:
        nIter = options["badmm_max_iters"]
    if "badmm_rho" in options:
        rho = options["badmm_rho"]*np.median(np.median(np.linalg.norm(c.supp.T - supp.T, axis=0)**2))
    else:
        rho = 2*np.mean(np.mean(np.linalg.norm(c.supp.T - supp.T, axis=0)**2))
    if "badmm_tau" in options:
        tau = options["tau"]
    else:
        tau = 10
    if "badmm_tol" in options:
        badmm_tol = options["badmm_tol"]
    else:
        badmm_tol = 1E-4
    for iter in range(nIter):
        # update X
        X = Z * np.exp((C+Y)/(-rho)) + 1e-8
        X = (X.T * w.T / np.sum(X, axis=1)).T
        
        # update Z
        Z0 = np.copy(Z)
        Z = X * np.exp(Y/rho) + 1e-8
