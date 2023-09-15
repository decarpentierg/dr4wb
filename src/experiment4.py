import numpy as np
import matplotlib.pyplot as plt
from utils import emd
from tqdm import tqdm
from gensim.models import KeyedVectors

class Experiment:
    def __init__(self, mu1, mu2, m) -> None:
        assert mu1.shape == mu2.shape
        self.n, self.d = mu1.shape
        self.mu1 = mu1
        self.mu2 = mu2
        self.m = m
        self.rng = np.random.default_rng(seed=42)

    def experiment(self, dim):
        """Project mu1 and mu2 and compare the resulting Wasserstein distance with the original distance."""
        mu1 = self.mu1[:, :dim]
        mu2 = self.mu2[:, :dim]
        gamma, costs = emd(mu1, mu2)
        true_cost = np.sum(gamma * costs)
        projection = self.rng.normal(size=(self.m, dim)) / np.sqrt(self.m)
        mu1_ = mu1 @ projection.T
        mu2_ = mu2 @ projection.T
        gamma_, _ = emd(mu1_, mu2_)
        return np.sum(gamma_ * costs) / true_cost

    def sweep_dim(self, dim_values, n_samples):
        quality_ratios = np.zeros((len(dim_values), n_samples))
        for dim_idx, dim in tqdm(enumerate(dim_values)):
            for k in range(n_samples):
                quality_ratios[dim_idx, k] = self.experiment(dim)
        return quality_ratios


def plot_quality_ratios(x, quality_ratios, label=""):
    y_mean = np.mean(quality_ratios, axis=1)
    y_std= np.std(quality_ratios, axis=1)
    y_err = 2 * y_std / np.sqrt(quality_ratios.shape[1])
    plt.errorbar(x, y_mean, yerr=y_err, label=label)
    plt.fill_between(x, y_mean - y_err, y_mean + y_err, alpha=0.3)


np.random.seed(0)
N, D, M = 100, 300, 30
MU1 = np.random.normal(size=(N, D))
MU2 = np.random.normal(size=(N, D))
offset = np.zeros(D)
offset[0] = 10
MU3 = MU2 + offset
exp1 = Experiment(MU1, MU2, M)
exp2 = Experiment(MU1, MU3, M)
DIM_VALUES = [3, 10, 30, 100, 300]
K = 30
qr1 = exp1.sweep_dim(DIM_VALUES, K)
qr2 = exp2.sweep_dim(DIM_VALUES, K)
# plot_quality_ratios(M_VALUES, qr1, title="Experiment 1")
# plot_quality_ratios(M_VALUES, qr2, title="Experiment 1")

MU4 = KeyedVectors.load_word2vec_format("data/wiki.en.align.10k.vec", binary=False).get_normed_vectors()[:100]
MU5 = KeyedVectors.load_word2vec_format("data/wiki.fr.align.10k.vec", binary=False).get_normed_vectors()[:100]
exp3 = Experiment(MU4, MU5, M)
qr3 = exp3.sweep_dim(DIM_VALUES, K)
