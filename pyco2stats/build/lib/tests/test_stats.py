import unittest
import numpy as np
from PyCO2stats import Stats

class TestStats(unittest.TestCase):
    def test_sample_from_pdf(self):
        x = np.linspace(0, 10, 100)
        pdf = np.exp(-x)
        n_samples = 1000
        
        samples = Stats.sample_from_pdf(x, pdf, n_samples)
        
        self.assertTrue(np.all(samples >= x[0]) and np.all(samples <= x[-1]))
        self.assertEqual(len(samples), n_samples)
        self.assertTrue(np.mean(samples) < 10 and np.mean(samples) > 0)

    def test_mvue_lnorm_dist(self):
        data = np.random.lognormal(mean=1.0, sigma=0.5, size=100)
        mvue_mean, mvue_ci_lower, mvue_ci_upper = Stats.mvue_lnorm_dist(data)
        
        self.assertTrue(np.isfinite(mvue_mean))
        self.assertTrue(np.isfinite(mvue_ci_lower))
        self.assertTrue(np.isfinite(mvue_ci_upper))
        self.assertTrue(mvue_ci_lower <= mvue_mean <= mvue_ci_upper)

    def test_sichel_function(self):
        sigma_sq = 0.5
        n = 100
        psi_n = Stats.sichel_function(sigma_sq, n)
        
        self.assertTrue(np.isfinite(psi_n))
        self.assertGreater(psi_n, 0)

    def test_sichel_function_15(self):
        z = 0.5
        n = 100
        sf_15 = Stats.sichel_function_15(z, n)
        
        self.assertTrue(np.isfinite(sf_15))
        self.assertGreater(sf_15, 0)

    def test_median(self):
        data = np.array([1, 2, 3, 4, 5])
        median = Stats.median(data)
        
        self.assertEqual(median, 3)

    def test_median_absolute_deviation(self):
        data = np.array([1, 2, 3, 4, 5])
        mad = Stats.median_absolute_deviation(data)
        
        self.assertEqual(mad, 1)

    def test_mad_std(self):
        data = np.array([1, 2, 3, 4, 5])
        mad_std = Stats.mad_std(data)
        
        self.assertAlmostEqual(mad_std, 1.4826, places=4)

    def test_sigma_clip(self):
        data = np.array([1, 2, 3, 4, 100])
        clipped_data = Stats.sigma_clip(data, sigma=2)
        
        self.assertTrue(np.all(clipped_data.mask == np.array([False, False, False, False, True])))

    def test_sigma_clipped_stats(self):
        data = np.array([1, 2, 3, 4, 100])
        mean, median, stddev = Stats.sigma_clipped_stats(data, sigma=2)
        
        self.assertAlmostEqual(mean, 2.5, places=1)
        self.assertAlmostEqual(median, 2.5, places=1)
        self.assertAlmostEqual(stddev, 1.290, places=3)

    def test_biweight_location(self):
        data = np.array([1, 2, 3, 4, 100])
        biweight_loc = Stats.biweight_location(data)
        
        self.assertAlmostEqual(biweight_loc, 2.982, places=3)

    def test_biweight_scale(self):
        data = np.array([1, 2, 3, 4, 100])
        biweight_scl = Stats.biweight_scale(data)
        
        self.assertAlmostEqual(biweight_scl, 1.349, places=3)

    def test_trim(self):
        data = np.array([1, 2, 3, 4, 100])
        trimmed_data = Stats.trim(data, limits=(0.2, 0.2))
        
        self.assertEqual(trimmed_data.count(), 3)

    def test_trimmed_mean(self):
        data = np.array([1, 2, 3, 4, 100])
        tmean = Stats.trimmed_mean(data, limits=(0.2, 0.2))
        
        self.assertAlmostEqual(tmean, 3.0)

    def test_trimmed_std(self):
        data = np.array([1, 2, 3, 4, 100])
        tstd = Stats.trimmed_std(data, limits=(0.2, 0.2))
        
        self.assertAlmostEqual(tstd, 1.0, places=1)

    def test_trimboth(self):
        data = np.array([1, 2, 3, 4, 100])
        trimboth_data = Stats.trimboth(data, proportiontocut=0.2)
        
        self.assertEqual(len(trimboth_data), 3)

    def test_trimtail(self):
        data = np.array([1, 2, 3, 4, 100])
        trimtail_data = Stats.trimtail(data, proportiontocut=0.2)
        
        self.assertEqual(trimtail_data.count(), 4)

    def test_winsorize(self):
        data = np.array([1, 2, 3, 4, 100])
        winsorized_data = Stats.winsorize(data, limits=(0.2, 0.2))
        
        self.assertTrue(np.all(winsorized_data == np.array([1, 2, 3, 4, 4])))

    def test_winsorized_mean(self):
        data = np.array([1, 2, 3, 4, 100])
        winsorized_mean = Stats.winsorized_mean(data, limits=(0.2, 0.2))
        
        self.assertAlmostEqual(winsorized_mean, 2.8)

    def test_winsorized_std(self):
        data = np.array([1, 2, 3, 4, 100])
        winsorized_std = Stats.winsorized_std(data, limits=(0.2, 0.2))
        
        self.assertAlmostEqual(winsorized_std, 1.3, places=1)

    def test_Huber(self):
        data = np.array([1, 2, 3, 4, 100])
        location, scale = Stats.Huber(data)
        
        self.assertAlmostEqual(location, 2.5)
        self.assertAlmostEqual(scale, 1.290, places=3)

if __name__ == '__main__':
    unittest.main()
