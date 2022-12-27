import numpy as np
import matplotlib.pyplot as plt
from utils import emd
from tqdm import tqdm

class Experiment:
    def __init__(self, mu1, mu2) -> None:
        assert mu1.shape == mu2.shape
        self.n, self.d = mu1.shape
        self.mu1 = mu1
        self.mu2 = mu2
        self.gamma, self.costs = emd(mu1, mu2)
        self.true_cost = np.sum(self.gamma * self.costs)
        self.rng = np.random.default_rng(seed=42)

    def experiment(self, m):
        """Project mu1 and mu2 and compare the resulting Wasserstein distance with the original distance."""
        projection = self.rng.normal(size=(m, self.d)) / np.sqrt(m)
        mu1_ = self.mu1 @ projection.T
        mu2_ = self.mu2 @ projection.T
        gamma, _ = emd(mu1_, mu2_)
        return np.sum(gamma * self.costs) / self.true_cost

    def sweep_projection_dimension(self, m_values, n_samples):
        quality_ratios = np.zeros((len(m_values), n_samples))
        for m_idx, m in tqdm(enumerate(m_values)):
            for k in range(n_samples):
                quality_ratios[m_idx, k] = self.experiment(m)
        return quality_ratios


def plot_quality_ratios(m_values, quality_ratios, label=""):
    y_mean = np.mean(quality_ratios, axis=1)
    y_std= np.std(quality_ratios, axis=1)
    y_err = 2 * y_std / np.sqrt(quality_ratios.shape[1])
    plt.errorbar(m_values, y_mean, yerr=y_err, label=label)
    plt.fill_between(m_values, y_mean - y_err, y_mean + y_err, alpha=0.3)


np.random.seed(0)
N, D = 100, 300
MU1 = np.random.normal(size=(N, D))
MU2 = np.random.normal(size=(N, D))
offset = np.zeros(D)
offset[0] = 10
MU3 = MU2 + offset
exp1 = Experiment(MU1, MU2)
exp2 = Experiment(MU1, MU3)
M_VALUES = [3, 10, 30, 100, 300]
K = 30
qr1 = exp1.sweep_projection_dimension(M_VALUES, K)
qr2 = exp2.sweep_projection_dimension(M_VALUES, K)
# plot_quality_ratios(M_VALUES, qr1, title="Experiment 1")
# plot_quality_ratios(M_VALUES, qr2, title="Experiment 1")
