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
        quality_ratios = np.zeros((len(m_values), k))
        for m_idx, m in tqdm(enumerate(m_values)):
            for k in range(n_samples):
                quality_ratios[m_idx, k] = self.experiment(m)
        return quality_ratios


def plot_quality_ratios(m_values, quality_ratios, title="Experiment 1", xlabel="Projection dimension m"):
    y_mean = np.mean(quality_ratios, axis=1)
    y_std= np.std(quality_ratios, axis=1)
    y_err = 2 * y_std / np.sqrt(quality_ratios.shape[1])

    plt.rcParams.update({'font.size': 22})
    plt.figure(figsize=(13, 8))
    plt.errorbar(m_values, y_mean, yerr=y_err)
    plt.fill_between(m_values, y_mean - y_err, y_mean + y_err, alpha=0.3)
    plt.grid()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Ratio of solution quality")
    plt.show()
