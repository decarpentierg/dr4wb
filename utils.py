import numpy as np
import ot

def emd(mu1, mu2):
    """Return EMD distance between mu1 and mu2, where mu1 and mu2 are two discrete distributions with uniform weights
    represented as nd-arrays of shape (n1, d) and (n2, d)."""
    costs = ot.dist(mu1, mu2)
    w1 = np.ones(mu1.shape[0])  # uniform weights
    w2 = np.ones(mu1.shape[1])
    gamma, log = ot.emd(w1, w2, costs, log=True)
    # check that everything went fine
    assert log["result_code"] == 1, "ot.emd encountered a problem"
    assert log["warning"] == "", "ot.emd encountered a problem"
    # return transportation matrix and transportation costs
    return gamma, costs  # shape (n1, n2)
