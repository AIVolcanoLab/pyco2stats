import numpy as np
from scipy.stats import norm

class Sinclair:
    """
    Implements transformations between cumulative probabilities
    and sigma-values (standard normal quantiles) for probability plots.
    """

    @staticmethod
    def cumulative_to_sigma(p: np.ndarray) -> np.ndarray:
        p_clipped = np.clip(p, 1e-10, 1 - 1e-10)
        return norm.ppf(p_clipped)

    @staticmethod
    def sigma_to_cumulative(sigma: np.ndarray) -> np.ndarray:
        return norm.cdf(sigma)

    @staticmethod
    def raw_data_to_sigma(raw_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sorted_data = np.sort(raw_data)
        p = np.linspace(0, 1, len(sorted_data), endpoint=False) + 0.5 / len(sorted_data)
        sigma_values = Sinclair.cumulative_to_sigma(p)
        return sigma_values, sorted_data

    @staticmethod
    def combine_gaussians(x: np.ndarray, means: np.ndarray, stds: np.ndarray, weights: np.ndarray) -> np.ndarray:
        y_comb = np.zeros_like(x)
        for mu, sigma, w in zip(means, stds, weights):
            y_comb += w * norm.cdf(x, loc=mu, scale=sigma)
        return y_comb