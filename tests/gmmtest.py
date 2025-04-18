import unittest
import numpy as np
from scipy.stats import norm
import torch
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
import warnings

from pyco2stats.gaussian_mixtures import GMM  # Adjust the import according to your module's structure

class TestGMM(unittest.TestCase):
    def setUp(self):
        self.data = np.random.normal(loc=0, scale=1, size=100)
        self.n_components = 3
        self.mean_constraints = [(-2, 2), (-2, 2), (-2, 2)]
        self.std_constraints = [(0.5, 2), (0.5, 2), (0.5, 2)]
        self.max_iter = 100
        self.tol = 1e-6
        self.n_epochs = 1000
        self.lr = 0.001
        self.verbose = False

    def test_gaussian_mixture_em(self):
        means, std_devs, weights, log_likelihoods = GMM.gaussian_mixture_em(
            self.data, self.n_components, max_iter=self.max_iter, tol=self.tol
        )
        self.assertEqual(len(means), self.n_components)
        self.assertEqual(len(std_devs), self.n_components)
        self.assertEqual(len(weights), self.n_components)
        self.assertGreater(len(log_likelihoods), 0)
        self.assertAlmostEqual(np.sum(weights), 1, places=5)

    def test_gaussian_mixture_sklearn(self):
        X = self.data.reshape(-1, 1)
        original_means, original_std_devs, weights, max_iter, log_likelihoods = GMM.gaussian_mixture_sklearn(
            X, n_components=self.n_components, max_iter=self.max_iter, tol=self.tol
        )
        self.assertEqual(len(original_means), self.n_components)
        self.assertEqual(len(original_std_devs), self.n_components)
        self.assertEqual(len(weights), self.n_components)
        self.assertGreater(len(log_likelihoods), 0)
        self.assertAlmostEqual(np.sum(weights), 1, places=5)

    def test_constrained_gaussian_mixture(self):
        optimized_means, optimized_stds, optimized_weights = GMM.constrained_gaussian_mixture(
            self.data, self.mean_constraints, self.std_constraints, self.n_components,
            n_epochs=self.n_epochs, lr=self.lr, verbose=self.verbose
        )
        self.assertEqual(len(optimized_means), self.n_components)
        self.assertEqual(len(optimized_stds), self.n_components)
        self.assertEqual(len(optimized_weights), self.n_components)
        self.assertAlmostEqual(np.sum(optimized_weights), 1, places=5)
        for i in range(self.n_components):
            self.assertGreaterEqual(optimized_means[i], self.mean_constraints[i][0])
            self.assertLessEqual(optimized_means[i], self.mean_constraints[i][1])
            self.assertGreaterEqual(optimized_stds[i], self.std_constraints[i][0])
            self.assertLessEqual(optimized_stds[i], self.std_constraints[i][1])

    def test_gaussian_mixture_pdf(self):
        x = np.linspace(-5, 5, 100)
        meds = [0, 1]
        stds = [1, 0.5]
        weights = [0.6, 0.4]
        pdf = GMM.gaussian_mixture_pdf(x, meds, stds, weights)
        self.assertEqual(len(pdf), len(x))
        self.assertTrue(np.all(pdf >= 0))
        self.assertAlmostEqual(np.sum(pdf) * (x[1] - x[0]), 1, places=1)

if __name__ == '__main__':
    unittest.main()
