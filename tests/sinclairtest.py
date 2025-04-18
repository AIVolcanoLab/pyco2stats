import unittest
import numpy as np
import scipy.stats
import pyco2stats
from scipy.stats import norm
from pyco2stats.sinclair import Sinclair  # Assuming your class is defined in sinclair.py

class TestSinclair(unittest.TestCase):

    def setUp(self):
        self.test_data = np.array([1, 2, 3, 4, 5])  # Example test data
        self.meds = [0, 1, 2]  # Example means for Gaussian components
        self.stds = [1, 1, 1]  # Example standard deviations for Gaussian components
        self.fds = [0.3, 0.5, 0.2]  # Example weights for Gaussian components
        self.mminy = 0  # Example minimum value for ylon
        self.mmaxy = 10  # Example maximum value for ylon
        self.n = 100  # Example number of points to generate

    def test_get_raw_data(self):
        osm, osr = Sinclair.get_raw_data(self.test_data)
        
        # Assertions
        self.assertEqual(osm.shape, self.test_data.shape, "osm should have the same shape as my_data")
        self.assertEqual(osr.shape, self.test_data.shape, "osr should have the same shape as my_data")
        self.assertTrue(np.all(np.isfinite(osm)), "osm should contain finite values")
        self.assertTrue(np.all(np.isfinite(osr)), "osr should contain finite values")

    def test_calculate_combined_population(self):
        ixe, ylon_1 = Sinclair.calculate_combined_population(self.meds, self.stds, self.fds, self.mminy, self.mmaxy, self.n)
        
        # Assertions
        self.assertEqual(ixe.shape[0], self.n, "ixe should have length equal to n")
        self.assertEqual(ylon_1.shape[0], self.n, "ylon_1 should have length equal to n")
        self.assertTrue(np.all(np.isfinite(ixe)), "ixe should contain finite values")
        self.assertTrue(np.all(np.isfinite(ylon_1)), "ylon_1 should contain finite values")

    def test_calc_uniform_order_statistic_medians(self):
        n = 5
        v = Sinclair._calc_uniform_order_statistic_medians(n)
        
        # Assertions
        self.assertEqual(v.shape, (n,), "v should have shape (n,)")
        self.assertTrue(np.all(np.isfinite(v)), "v should contain finite values")
    
    # Add more tests as needed

if __name__ == '__main__':
    unittest.main()

