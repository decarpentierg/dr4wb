import numpy as np
from utils import emd
from tqdm import tqdm

np.random.seed(0)

N = 100  # nb of samples in each distribution
D = 300  # initial dimension

# distributions
MU1 = np.random.normal(size=(N, D))
MU2 = np.random.normal(size=(N, D))
GAMMA, COSTS = emd(MU1, MU2)
TRUE_COST = np.sum(GAMMA * COSTS)

M_VALUES = [3, 10, 30, 100, 300]
K = 30

def experiment(m):
    """Project mu1 and mu2 and compare the resulting Wasserstein distance with the original distance."""
    projection = np.random.normal(size=(m, D)) / np.sqrt(m)
    mu1_ = MU1 @ projection.T
    mu2_ = MU2 @ projection.T
    gamma, _ = emd(mu1_, mu2_)
    return np.sum(gamma * COSTS)


def sweep_projection_dimension():
    total_costs = np.zeros((len(M_VALUES), K))
    for m_idx, m in tqdm(enumerate(M_VALUES)):
        for k in range(K):
            total_costs[m_idx, k] = experiment(m)
    return total_costs


if __name__ == "__main__":
    total_costs = sweep_projection_dimension()
    np.savez_compressed("results/exp1_3.npz", m_values=np.array(M_VALUES), total_costs=total_costs, true_cost=TRUE_COST)
