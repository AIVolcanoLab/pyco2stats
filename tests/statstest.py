import unittest
from pyco2stats.stats import Stats as stats
import numpy as np
import scipy.special as sp
import scipy.stats
import statsmodels.api as sm

from scipy.stats.mstats import scoreatpercentile as scipy_scoreatpercentile
from scipy.stats.mstats import trim as scipy_trim
from scipy.stats import trimboth as scipy_trimboth
from scipy.stats.mstats import trimtail as scipy_trimtail
from scipy.stats import tmean as scipy_tmean
from scipy.stats.mstats import trimmed_std as scipy_trimmed_std 
from scipy.stats.mstats import winsorize as scipy_winsorize

from astropy.stats import biweight_location as astropy_biweight_location
from astropy.stats import biweight_scale as astropy_biweight_scale
from astropy.stats import median_absolute_deviation as astropy_median_absolute_deviation
from astropy.stats import mad_std as astropy_mad_std
from astropy.stats import sigma_clip as astropy_sigma_clip
from astropy.stats import sigma_clipped_stats as astropy_sigma_clipped_stats

class TestStats(unittest.TestCase):

    def test_sample_from_pdf(self):
        # Test data
        x = np.linspace(0, 10, 100)
        pdf = scipy.stats.norm.pdf(x, loc=5, scale=1)  # Normal distribution centered at 5

        # Number of samples to generate
        n_samples = 1000

        # Call the method
        samples = stats.sample_from_pdf(x, pdf, n_samples)

        # Check the number of samples generated
        self.assertEqual(len(samples), n_samples)

        # Check if samples are within the x range
        self.assertTrue(np.all(samples >= x.min()) and np.all(samples <= x.max()))

    def test_mvue_lnorm_dist(self):
        # Test data (log-normal distribution)
        np.random.seed(0)
        data = np.random.lognormal(mean=0, sigma=1, size=100)

        # Call the method
        mvue_mean, mvue_ci_lower, mvue_ci_upper = stats.mvue_lnorm_dist(data)

        # Check the results (basic sanity checks)
        self.assertTrue(mvue_mean > 0)
        self.assertTrue(mvue_ci_lower > 0)
        self.assertTrue(mvue_ci_upper > 0)
        self.assertTrue(mvue_ci_lower < mvue_mean < mvue_ci_upper)

    def test_sichel_function(self):
        # Test data
        z = 0.5
        n = 10
        M = 15

        # Call the method
        result = stats.sichel_function(z, n, M)

        # Basic sanity check
        self.assertTrue(result > 0)

    def test_sichel_function_log(self):
        # Test data
        sigma_sq = 0.25
        n = 10

        # Call the method
        result = stats.sichel_function_log(sigma_sq, n)
        

        # Basic sanity check
        self.assertTrue(result > 0)

    def test_median(self):
        data = np.array([1, 3, 5, 7, 9])
        result = stats.median(data)
        expected = 5
        self.assertEqual(result, expected)

    def test_median_with_axis(self):
        data = np.array([[1, 3, 5], [2, 4, 6]])
        result = stats.median(data, axis=0)
        expected = np.array([1.5, 3.5, 5.5])
        np.testing.assert_array_equal(result, expected)
   
    def test_median_absolute_deviation(self):
        data = np.array([1, 1, 2, 2, 4, 6, 9])
        result = stats.median_absolute_deviation(data)
        expected = 1  # Calculated manually: median(abs(data - median(data))) = median([0, 0, 1, 1, 1, 2, 3]) = 1.5
        self.assertEqual(result, expected)

    def test_mad_std(self):
        data = np.array([1, 1, 2, 2, 4, 6, 9])
        result = stats.mad_std(data)
        expected = 1.4826 * 1  # 1.4826 * MAD
        self.assertAlmostEqual(result, expected, places=4)
        
    def test_mad_std_with_nan(self):
        data = np.array([1, 1, 2, 2, np.nan, 6, 9])
        result = stats.mad_std(data, ignore_nan=True)
        expected = 1.4826 * 1  # 1.4826 * MAD without considering nan
        self.assertAlmostEqual(result, expected, places=4)

    def test_sigma_clip(self):
        data = np.array([1, 2, 3, 4, 5, 100])
        result = stats.sigma_clip(data, sigma=2)
        expected = np.ma.masked_array([1, 2, 3, 4, 5, 100], mask=[False, False, False, False, False, True])
        np.testing.assert_array_equal(result, expected)

    def test_sigma_clipped_stats(self):
        data = np.array([1, 2, 3, 4, 5])
        mean, median, stddev = stats.sigma_clipped_stats(data, sigma=2)
        expected_mean, expected_median, expected_stddev = 3, 3, 1.41421356  # Manually calculated
        self.assertAlmostEqual(mean, expected_mean)
        self.assertAlmostEqual(median, expected_median)
        self.assertAlmostEqual(stddev, expected_stddev, places=4)

    def test_biweight_location(self):
        data = np.array([1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 100])
        result = stats.biweight_location(data)
        expected = 2  # Manually calculated
        self.assertAlmostEqual(result, expected)

    def test_biweight_location_with_nan(self):
        data = np.array([1, 2, 2, 2, 2, np.nan, 2, 2, 2, 3, 100])
        result = stats.biweight_location(data, ignore_nan=True)
        expected = 2  # Manually calculated
        self.assertAlmostEqual(result, expected)

    def setUp(self):
        self.data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.masked_data = np.ma.array([1, 2, 3, np.nan, 5, 6, 7, 8, 9, 10], mask=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        self.data_with_nan = np.array([1, 2, 3, np.nan, 5, 6, 7, 8, 9, 10])
        
    def test_biweight_scale(self):
        result = stats.biweight_scale(self.data)
        expected = astropy_biweight_scale(self.data)
        np.testing.assert_almost_equal(result, expected, decimal=5)

        result = stats.biweight_scale(self.masked_data, ignore_nan=True)
        expected = astropy_biweight_scale(np.ma.masked_invalid(self.masked_data))
        np.testing.assert_almost_equal(result, expected, decimal=5)

    def test_trim(self):
        result = stats.trim(self.data, limits=(0.1, 0.1))
        expected = scipy_trim(self.data, limits=(0.1, 0.1))
        np.testing.assert_array_equal(result.mask, expected.mask)

    # error here! ValueError: No array values within given limits
    def test_trimmed_mean(self):
        result = stats.trimmed_mean(self.data, limits=(0.1, 0.9))
        expected = scipy_trimmed_mean(self.data, limits=(0.1, 0.9))
        self.assertAlmostEqual(result, expected, places=5)

    def test_trimmed_std(self):
        result = stats.trimmed_std(self.data, limits=(0.1, 0.1))
        expected = scipy_trimmed_std(self.data, limits=(0.1, 0.1))
        self.assertAlmostEqual(result, expected, places=5)

    def test_trimboth(self):
        result = stats.trimboth(self.data, proportiontocut=0.2)
        expected = scipy_trimboth(self.data, proportiontocut=0.2)
        np.testing.assert_array_equal(result, expected)

    def test_trimtail(self):
        result = stats.trimtail(self.data, proportiontocut=0.2, tail='left')
        expected = scipy_trimtail(self.data, proportiontocut=0.2, tail='left')
        np.testing.assert_array_equal(result.mask, expected.mask)

        result = stats.trimtail(self.data, proportiontocut=0.2, tail='right')
        expected = scipy_trimtail(self.data, proportiontocut=0.2, tail='right')
        np.testing.assert_array_equal(result.mask, expected.mask)

    def test_winsorize(self):
        result = stats.winsorize(self.data, limits=(0.1, 0.1))
        expected = scipy_winsorize(self.data, limits=(0.1, 0.1))
        np.testing.assert_array_equal(result, expected)

    def test_winsorized_mean(self):
        result = stats.winsorized_mean(self.data, limits=(0.1, 0.1))
        expected = np.mean(scipy_winsorize(self.data, limits=(0.1, 0.1)))
        self.assertAlmostEqual(result, expected, places=5)

    def test_winsorized_std(self):
        result = stats.winsorized_std(self.data, limits=(0.1, 0.1))
        expected = np.std(scipy_winsorize(self.data, limits=(0.1, 0.1)), ddof=1)
        self.assertAlmostEqual(result, expected, places=5)

    def test_Huber(self):
        huber_estimator = sm.robust.Huber()
        expected_location, expected_scale = huber_estimator(self.data)

        result_location, result_scale = stats.Huber(self.data)

        self.assertAlmostEqual(result_location, expected_location, places=5)
        self.assertAlmostEqual(result_scale, expected_scale, places=5)

if __name__ == '__main__':
    unittest.main()
