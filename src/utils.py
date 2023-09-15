import numpy as np
import matplotlib.pyplot as plt
import ot

def emd(mu1, mu2):
    """Return EMD distance between mu1 and mu2, where mu1 and mu2 are two discrete distributions with uniform weights
    represented as nd-arrays of shape (n1, d) and (n2, d)."""
    costs = ot.dist(mu1, mu2)
    w1 = np.ones(mu1.shape[0])  # uniform weights
    w2 = np.ones(mu2.shape[0])
    gamma, log = ot.emd(w1, w2, costs, log=True)
    # check that everything went fine
    assert log["result_code"] == 1, "ot.emd encountered a problem"
    assert log["warning"] is None, "ot.emd encountered a problem"
    # return transportation matrix and transportation costs
    return gamma, costs  # shape (n1, n2)


def plot2D(mu1, mu2, gamma):
    ot.plot.plot2D_samples_mat(mu1[:, :2], mu2[:, :2], gamma)
    plt.scatter(mu1[:, 0], mu1[:, 1])
    plt.scatter(mu2[:, 0], mu2[:, 1])
    plt.show()
