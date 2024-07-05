import numpy as np
import matplotlib.pyplot as plt
import os

"""
Implementation of Principal Component Analysis.
"""


class PCA:
    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.mean = None
        self.components = None

    def fit(self, X: np.ndarray) -> None:
        # TODO: 10%

        self.mean = np.mean(X, axis=0)
        X = X - self.mean
        c = X.T @ X
        eig_val, eig_vec = np.linalg.eig(c)

        eig_val = eig_val.real
        eig_vec = eig_vec.real

        idx = np.argsort(eig_val)[::-1]
        eig_val = eig_val[idx]
        eig_vec = eig_vec[:, idx]

        self.components = eig_vec[:, :self.n_components]

    def transform(self, X: np.ndarray) -> np.ndarray:
        # TODO: 2%

        X = X - self.mean
        return X @ self.components

    def reconstruct(self, X):
        # TODO: 2%

        return (self.transform(X) @ self.components.T) + self.mean

    def problem_a(self):
        plt.subplot(231)
        plt.imshow(np.reshape(self.mean, (61, 80)))
        plt.subplot(232)
        plt.imshow(np.reshape(self.components[:, 0], (61, 80)))
        plt.subplot(233)
        plt.imshow(np.reshape(self.components[:, 1], (61, 80)))
        plt.subplot(234)
        plt.imshow(np.reshape(self.components[:, 2], (61, 80)))
        plt.subplot(235)
        plt.imshow(np.reshape(self.components[:, 3], (61, 80)))

        if not os.path.exists('image'):
            os.makedirs('image')

        plt.savefig('image/problem_a.png')
        plt.clf()
