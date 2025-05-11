import numpy as np
from scipy.stats import norm

class Sinclair:
    """
    Implements transformations between cumulative probabilities
    and sigma-values (standard normal quantiles) for probability plots.
    """

    @staticmethod
    def cumulative_to_sigma(p: np.ndarray) -> np.ndarray:
        """
        Converts cumulative probabilities to sigma-values (z-scores).

        Parameters
        ----------
        p : array
            Array of cumulative probabilities in the range [0, 1].

        Returns
        -------
        sigma_values : array
            Array of sigma-values corresponding to the input probabilities.
        """
        p_clipped = np.clip(p, 1e-10, 1 - 1e-10)
        return norm.ppf(p_clipped)

    @staticmethod
    def sigma_to_cumulative(sigma: np.ndarray) -> np.ndarray:
        """
        Converts sigma-values (z-scores) to cumulative probabilities.

        Parameters
        ----------
        sigma : array
            Array of sigma-values.

        Returns
        -------
        cumulative_probs : array
            Array of cumulative probabilities corresponding to the input sigma-values.
        """
        return norm.cdf(sigma)

    @staticmethod
    def raw_data_to_sigma(raw_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Converts raw data into sorted values and their corresponding sigma-values.

        Parameters
        ----------
        raw_data : array
            Array of raw data values.

        Returns
        -------
        sigma_values : array
            Sigma-values corresponding to the empirical cumulative probabilities.
        sorted_data : array
            Raw data sorted in ascending order.
        """
        sorted_data = np.sort(raw_data)
        p = np.linspace(0, 1, len(sorted_data), endpoint=False) + 0.5 / len(sorted_data)
        sigma_values = Sinclair.cumulative_to_sigma(p)
        return sigma_values, sorted_data

    @staticmethod
    def combine_gaussians(x: np.ndarray, means: np.ndarray, stds: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Computes the cumulative distribution of a weighted mixture of Gaussian distributions.

        Parameters
        ----------
        x : array
            Points at which to evaluate the combined cumulative distribution.
        means : array
            Means of the individual Gaussian components.
        stds : array
            Standard deviations of the Gaussian components.
        weights : array
            Weights for each Gaussian component.

        Returns
        -------
        y_comb : array
            The combined cumulative distribution evaluated at the points x.
        """
        y_comb = np.zeros_like(x)
        for mu, sigma, w in zip(means, stds, weights):
            y_comb += w * norm.cdf(x, loc=mu, scale=sigma)
        return y_comb
