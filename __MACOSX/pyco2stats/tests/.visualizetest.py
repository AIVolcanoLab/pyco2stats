import unittest
import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for rendering plots without a display
import matplotlib.pyplot as plt
from unittest.mock import MagicMock

from pyco2stats.visualize import Visualize
from pyco2stats.sinclair import Sinclair
from pyco2stats.gaussian_mixtures import GMM

class TestVisualize(unittest.TestCase):
    
    def setUp(self):
        self.data = np.random.normal(loc=0, scale=1, size=100)
        self.meds = [0, 1]
        self.stds = [1, 0.5]
        self.weights = [0.6, 0.4]
        self.mminy = -3
        self.mmaxy = 3

    def test_pp_raw_data(self):
        fig, ax = plt.subplots()
        ax.scatter = MagicMock()  # Mock the scatter method before calling pp_raw_data
        Visualize.pp_raw_data(self.data, ax)
        self.assertTrue(ax.scatter.called)

    def test_pp_combined_population(self):
        fig, ax = plt.subplots()
        ax.plot = MagicMock()  # Mock the plot method before calling pp_combined_population
        Visualize.pp_combined_population(self.meds, self.stds, self.weights, self.mminy, self.mmaxy, ax)
        self.assertTrue(ax.plot.called)

    def test_pp_single_populations(self):
        fig, ax = plt.subplots()
        ax.plot = MagicMock()  # Mock the plot method before calling pp_single_populations
        Visualize.pp_single_populations(self.meds, self.stds, self.mminy, self.mmaxy, ax)
        self.assertTrue(ax.plot.called)

    def test_pp_add_percentiles(self):
        fig, ax = plt.subplots()
        ax.axvline = MagicMock()  # Mock the axvline method before calling pp_add_percentiles
        Visualize.pp_add_percentiles(ax)
        self.assertTrue(ax.axvline.called)

    def test_qq_plot(self):
        reference_population = np.random.normal(loc=0, scale=1, size=100)
        fig, ax = plt.subplots()
        ax.plot = MagicMock()  # Mock the plot method before calling qq_plot
        Visualize.qq_plot(ax, self.data, reference_population)
        self.assertTrue(ax.plot.called)

    def test_plot_gmm_pdf(self):
        x = np.linspace(-5, 5, 100)
        fig, ax = plt.subplots()
        ax.plot = MagicMock()  # Mock the plot method before calling plot_gmm_pdf
        ax.hist = MagicMock()  # Mock the hist method before calling plot_gmm_pdf
        Visualize.plot_gmm_pdf(ax, x, self.meds, self.stds, self.weights, data=self.data)
        self.assertTrue(ax.plot.called)
        self.assertTrue(ax.hist.called)

if __name__ == '__main__':
    unittest.main()
