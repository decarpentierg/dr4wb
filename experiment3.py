import numpy as np
import matplotlib.pyplot as plt
import ot
from gensim.models import KeyedVectors

class Experiment:
    def __init__(self, mu1, mu2, m) -> None:
        assert mu1.shape == mu2.shape
        self.n, self.d = mu1.shape
        self.mu1 = mu1
        self.mu2 = mu2
        self.m = m
        self.rng = np.random.default_rng()

    def experiment(self):
        """Project mu1 and mu2 and compare the resulting Wasserstein distance with the original distance."""
        projection = self.rng.normal(size=(self.m, self.d)) / np.sqrt(self.m)
        mu1_ = self.mu1 @ projection.T
        mu2_ = self.mu2 @ projection.T
        costs = ot.dist(self.mu1, self.mu2)
        costs_ = ot.dist(mu1_, mu2_)
        return costs_ / costs


# np.random.seed(0)
N, D = 1000, 300
MU1 = np.random.normal(size=(N, D))
MU2 = np.random.normal(size=(N, D))
offset = np.zeros(D)
offset[0] = 10
MU3 = MU2 + offset
MU4 = KeyedVectors.load_word2vec_format("data/wiki.en.align.10k.vec", binary=False).get_normed_vectors()[:1000]
MU5 = KeyedVectors.load_word2vec_format("data/wiki.fr.align.10k.vec", binary=False).get_normed_vectors()[:1000]
MU6 = np.random.randint(2, size=(N, D)) * 2. - 1
MU7 = np.random.randint(2, size=(N, D)) * 2. - 1
MU8 = np.random.normal(size=(N, D))
MU8[:N // 2, 0] += 10 
MU9 = np.random.normal(size=(N, D))
MU9[:N // 2, 0] += 10 

def plot(m):
    exp1 = Experiment(MU1, MU2, m)
    exp2 = Experiment(MU1, MU3, m)
    exp3 = Experiment(MU4, MU5, m)

    dr1 = exp1.experiment()
    dr2 = exp2.experiment()
    dr3 = exp3.experiment()

    plt.rcParams.update({'font.size': 22})
    plt.figure(figsize=(13, 8))
    colors = ["blue", "red", "green", "yellow"]
    h1 = plt.hist(dr1.flatten(), bins=50, alpha=0.6, density=True, color=colors[0], label="offset=0")
    h2 = plt.hist(dr2.flatten(), bins=50, alpha=0.6, density=True, color=colors[1], label="offset=10")
    h3 = plt.hist(dr3.flatten(), bins=50, alpha=0.6, density=True, color=colors[2], label="word vectors")

    for h_idx, h in enumerate([h1, h2, h3]):
        plt.plot((h[1][:-1] + h[1][1:]) / 2, h[0], color=colors[h_idx])
    plt.grid()
    plt.legend()
    plt.xlabel("Distance in high dim / distance in low dim")
    plt.ylabel("Frequency")
    plt.title(f"m={m}")
    plt.savefig(f"figures/experiment_3_m={m}.png")
    plt.show()
