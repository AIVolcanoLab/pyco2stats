import numpy as np
import scipy.special as sp
import scipy.stats as stats
import statsmodels.api as sm
import warnings
import math

from scipy.interpolate import RegularGridInterpolator

from scipy.stats.mstats import trim as scipy_trim
from scipy.stats import trimboth as scipy_trimboth
from scipy.stats.mstats import trimtail as scipy_trimtail
from scipy.stats import tmean as scipy_tmean
from scipy.stats.mstats import trimmed_std as scipy_trimmed_std 
from scipy.stats.mstats import winsorize as scipy_winsorize
from scipy.special import hyp0f1
import scipy.interpolate
from scipy.stats import norm, t, chi2

from astropy.stats import biweight_location as astropy_biweight_location
from astropy.stats import biweight_scale as astropy_biweight_scale
from astropy.stats import median_absolute_deviation as astropy_median_absolute_deviation
from astropy.stats import mad_std as astropy_mad_std
from astropy.stats import sigma_clip as astropy_sigma_clip
from astropy.stats import sigma_clipped_stats as astropy_sigma_clipped_stats

from typing import List, Union, Dict, Any, Optional, Sequence
from scipy.stats import t as t_distribution # da mettere a posto t è richiamata due volte in modo diverso
from scipy.stats import norm as norm_distribution # da mettere a posto norm è richiamata due volte in modo diverso

class Stats:
    @staticmethod
    def umvu_lognormal_mean_variance(data, tol=1e-9, max_iter=1000):
        """
        Calculates the Uniformly Minimum Variance Unbiased (UMVU) estimators
        for the mean and variance of a two-parameter log-normal distribution
        based on the formulas derived by Finney (1941), referenced in Zhou and Gao (1997).
        Uses scipy.special.hyp0f1 for the generalized hypergeometric function _0F1.

        Reference:
            Zhou, X-H., & Gao, S. (1997). Confidence intervals for the log-normal mean.
            Statistics in Medicine, 16(7), 783-790. (Section 3)
            Finney, D. J. (1941). On the distribution of a variate whose logarithm is
            normally distributed. Supplement to the Journal of the Royal Statistical Society,
            7(2), 155-161.

        Parameters:
        ----------
        data : array-like
            A 1D array or list containing the log-normally distributed data (on the original scale).
            Must contain only positive values.
        tol : float, optional
            Tolerance for the convergence of the _0F1 infinite series used in the estimators.
            Note: This is primarily for reference if using a custom implementation.
            Defaults to 1e-9.
        max_iter : int, optional
            Maximum number of terms to sum in the _0F1 infinite series.
            Note: This is primarily for reference if using a custom implementation.
            Defaults to 1000.

        Returns:
        -------
        tuple
            A tuple containing:
                   - umvu_mean (float): The UMVU estimator of the log-normal mean.
                                        Returns NaN if data is not positive or sample size < 1.
                   - umvu_variance (float): The UMVU estimator of the log-normal variance.
                                            Returns NaN if data is not positive or sample size < 3.
        """
        data = np.asarray(data)
        n = len(data)

        # Check for positive values
        if np.any(data <= 0):
            warnings.warn("Data contains non-positive values. Lognormal distribution requires positive data.", UserWarning)
            return np.nan, np.nan

        # Handle small sample sizes
        if n < 1:
             return np.nan, np.nan
        if n == 1:
             # For n=1, the sample value is the UMVU estimator of the mean. Variance is undefined.
             return float(data[0]), np.nan # Return as float for consistency

        # Calculate parameters of the underlying normal distribution
        log_data = np.log(data)
        y_bar = np.mean(log_data)
        # Unbiased sample variance of the log-transformed data
        s_sq = np.var(log_data, ddof=1) # ddof=1 provides (n-1) in the denominator

        # --- Calculate UMVU Mean Estimator ---
        # Formula: exp(y_bar) * _0F1( (n-1)/2; (n-1)^2 / (4n) * s_sq )
        alpha_mean = (n - 1.0) / 2.0
        z_mean = (n - 1.0)**2 / (4.0 * n) * s_sq

        # Use scipy.special.hyp0f1 for _0F1
        try:
            phi_mean_0F1 = hyp0f1(alpha_mean, z_mean)
        except Exception as e:
            warnings.warn(f"Error computing hyp0f1 for mean: {e}", RuntimeWarning)
            phi_mean_0F1 = np.nan

        if not np.isnan(phi_mean_0F1):
            umvu_mean = np.exp(y_bar) * phi_mean_0F1
        else:
            umvu_mean = np.nan


        # --- Calculate UMVU Variance Estimator ---
        # Formula: exp(2 * y_bar) * [_0F1( (n-1)/2; (n-1)^2 / n * s_sq ) - _0F1( (n-1)/2; (n-1)(n-2) / (2n) * s_sq )]
        # This estimator is valid for n > 2.
        umvu_variance = np.nan # Initialize as NaN for n <= 2

        if n > 2:
            alpha_var = (n - 1.0) / 2.0 # Same alpha as for mean
            z_var1 = (n - 1.0)**2 / n * s_sq
            z_var2 = (n - 1.0) * (n - 2.0) / (2.0 * n) * s_sq

            # Use scipy.special.hyp0f1 for _0F1 terms
            try:
                phi_var1_0F1 = hyp0f1(alpha_var, z_var1)
                phi_var2_0F1 = hyp0f1(alpha_var, z_var2)
            except Exception as e:
                warnings.warn(f"Error computing hyp0f1 for variance: {e}", RuntimeWarning)
                phi_var1_0F1 = np.nan
                phi_var2_0F1 = np.nan


            # Handle potential NaN from hyp0f1
            if not np.isnan(phi_var1_0F1) and not np.isnan(phi_var2_0F1):
                 umvu_variance = np.exp(2 * y_bar) * (phi_var1_0F1 - phi_var2_0F1)
            else:
                 umvu_variance = np.nan


        return umvu_mean, umvu_variance

    @staticmethod
    def lognormal_mean_naive_ci(data, confidence_level=0.95):
        """
        Computes the Uniformly Minimum Variance Unbiased (UMVU) estimate of the log-normal mean
        and its confidence interval using the Naive method.

        This CI method is based on the standard error of the mean of the log-transformed data.
        As noted in Zhou and Gao (1997), this method is generally inappropriate for
        constructing confidence intervals for the log-normal mean, especially for small
        sample sizes, as it tends to undercover the true mean.

        Reference:
            Zhou, X-H., & Gao, S. (1997). Confidence intervals for the log-normal mean.
            Statistics in Medicine, 16(7), 783-790. (Formula 1)

        Parameters:
        ----------
        data : array-like
            A 1D array or list containing the log-normally distributed data (on the original scale).
            Must contain only positive values.
        confidence_level : float, optional
            The desired confidence level (e.g., 0.95 for 95%). Must be between 0 and 1.
            Default is 0.95.

        Returns:
        -------
        tuple
            A tuple containing:
                   - umvu_mean (float): The UMVU estimator of the log-normal mean.
                                        Returns NaN if data is not positive or sample size < 1,
                                        or if UMVU calculation fails.
                   - lower_bound (float): The lower bound of the confidence interval on the original scale.
                                          Returns NaN if data is not positive, sample size < 2, or confidence_level is invalid.
                   - upper_bound (float): The upper bound of the confidence interval on the original scale.
                                          Returns NaN if data is not positive, sample size < 2, or confidence_level is invalid.
            Returns (np.nan, np.nan, np.nan) if basic requirements (positive data, n>=1 for UMVU, n>=2 for CI) are not met or UMVU fails.
        """
        data = np.asarray(data)
        n = len(data)

        # Calculate the UMVU mean estimate first. It handles n=1 and positive data checks.
        umvu_mean, _ = Stats.umvu_lognormal_mean_variance(data) # UMVU variance not needed for the CI


        # Check if basic requirements for CI calculation are met (positive data already checked by umvu_lognormal_estimators if called with n>=1)
        if np.any(data <= 0) or n < 2 or not 0 < confidence_level < 1:
             if n < 1 or np.any(data <= 0):
                 warnings.warn("Data contains non-positive values or sample size < 1. Cannot calculate UMVU or CI.", UserWarning)
             elif n < 2:
                 warnings.warn("Sample size must be at least 2 for confidence interval calculation.", UserWarning)
             elif not 0 < confidence_level < 1:
                 warnings.warn("Confidence level must be between 0 and 1.", UserWarning)

             # Return NaN for CI bounds if calculation is not possible, but return UMVU if it was calculated
             return umvu_mean, np.nan, np.nan


        # Calculate parameters of the underlying normal distribution
        log_data = np.log(data)
        y_bar = np.mean(log_data)
        s_sq = np.var(log_data, ddof=1) # Unbiased sample variance of the log-transformed data

        # Calculate the standard error of y_bar
        se_y_bar = np.sqrt(s_sq / n)

        # Find the critical value from the standard normal distribution (z-score)
        alpha = 1 - confidence_level
        z_critical = norm.ppf(1 - alpha / 2)

        # Calculate the confidence interval bounds using Formula 1 from Zhou & Gao (1997)
        # Formula 1: exp(y_bar) * exp( +/- z * sqrt(s_y^2/n) )
        lower_bound = np.exp(y_bar) * np.exp(-z_critical * se_y_bar)
        upper_bound = np.exp(y_bar) * np.exp(z_critical * se_y_bar)

        # UMVU mean was already calculated


        return umvu_mean, lower_bound, upper_bound

    @staticmethod
    def lognormal_mean_cox_ci(data, confidence_level=0.95):
        """
        Computes the Uniformly Minimum Variance Unbiased (UMVU) estimate of the log-normal mean
        and its confidence interval using Cox's method.

        This CI method is based on likelihood ratio statistics and uses a t-distribution.
        Zhou and Gao (1997) found that Cox's method generally has the smallest coverage
        error for moderate and large sample sizes.

        Reference:
            Zhou, X-H., & Gao, S. (1997). Confidence intervals for the log-normal mean.
            Statistics in Medicine, 16(7), 783-790. (Formula 2)

        Parameters:
        ----------
        data : array-like
            A 1D array or list containing the log-normally distributed data (on the original scale).
            Must contain only positive values.
        confidence_level : float, optional
            The desired confidence level (e.g., 0.95 for 95%). Must be between 0 and 1.
            Default is 0.95.

        Returns:
        -------
        tuple
            A tuple containing:
                   - umvu_mean (float): The UMVU estimator of the log-normal mean.
                                        Returns NaN if data is not positive or sample size < 1,
                                        or if UMVU calculation fails.
                   - lower_bound (float): The lower bound of the confidence interval on the original scale.
                                          Returns NaN if data is not positive, sample size < 2, or confidence_level is invalid.
                   - upper_bound (float): The upper bound of the confidence interval on the original scale.
                                          Returns NaN if data is not positive, sample size < 2, or confidence_level is invalid.
            Returns (np.nan, np.nan, np.nan) if basic requirements (positive data, n>=1 for UMVU, n>=2 for CI) are not met or UMVU fails.
        """
        data = np.asarray(data)
        n = len(data)

        # Calculate the UMVU mean estimate first. It handles n=1 and positive data checks.
        umvu_mean, _ = Stats.umvu_lognormal_mean_variance(data) # UMVU variance not needed for the CI


        # Check if basic requirements for CI calculation are met (positive data already checked by umvu_lognormal_estimators if called with n>=1)
        if np.any(data <= 0) or n < 2 or not 0 < confidence_level < 1:
             if n < 1 or np.any(data <= 0):
                 warnings.warn("Data contains non-positive values or sample size < 1. Cannot calculate UMVU or CI.", UserWarning)
             elif n < 2:
                 warnings.warn("Sample size must be at least 2 for confidence interval calculation.", UserWarning)
             elif not 0 < confidence_level < 1:
                 warnings.warn("Confidence level must be between 0 and 1.", UserWarning)

             # Return NaN for CI bounds if calculation is not possible, but return UMVU if it was calculated
             return umvu_mean, np.nan, np.nan

        # Calculate parameters of the underlying normal distribution
        log_data = np.log(data)
        y_bar = np.mean(log_data)
        s_sq = np.var(log_data, ddof=1) # Unbiased sample variance of the log-transformed data

        # Degrees of freedom for the t-distribution
        df = n - 1

        # Find the critical value from the t-distribution
        alpha = 1 - confidence_level
        t_critical = t.ppf(1 - alpha / 2, df)

        # Calculate the confidence interval bounds using Formula 2 from Zhou & Gao (1997)
        # The formula is: exp(y_bar + s_y^2/2) * [1 +/- t_{n-1, 1-alpha/2} * sqrt((exp(s_y^2)-1)/n)]
        exp_s_sq = np.exp(s_sq)
        sqrt_term_arg = (exp_s_sq - 1.0) / n

        # Ensure the argument for sqrt is non-negative
        if sqrt_term_arg < 0:
             warnings.warn("Argument for sqrt in Cox's CI calculation is negative. Confidence interval may not be calculable.", RuntimeWarning)
             return umvu_mean, np.nan, np.nan # Return UMVU if calculated, but NaN for CI


        sqrt_term = np.sqrt(sqrt_term_arg)


        # Ensure term inside exp is finite
        exp_arg = y_bar + s_sq / 2.0
        if not np.isfinite(exp_arg):
             warnings.warn("Argument for exp in Cox's CI calculation is not finite.", RuntimeWarning)
             return umvu_mean, np.nan, np.nan # Return UMVU if calculated, but NaN for CI


        lower_bound_factor = 1.0 - t_critical * sqrt_term
        upper_bound_factor = 1.0 + t_critical * sqrt_term

        # Ensure factors are non-negative for meaningful bounds on original scale
        lower_bound = np.exp(exp_arg) * np.maximum(lower_bound_factor, 0)
        upper_bound = np.exp(exp_arg) * np.maximum(upper_bound_factor, 0)


        # UMVU mean was already calculated

        return umvu_mean, lower_bound, upper_bound


    @staticmethod
    def angus_conservative_lognormal_ci(data, confidence_level=0.95):
      """
      Calculates the confidence interval for the mean of a lognormal distribution
      using Angus's conservative method based on Zhou & Gao (1997).

      Args:
        data: A list, numpy array, or similar iterable of positive numerical data
              assumed to follow a lognormal distribution.
        confidence_level: The desired confidence level (e.g., 0.95 for 95%).

      Returns:
        A tuple containing the lower and upper bounds of the confidence interval
        for the mean (theta) of the lognormal distribution.

      Raises:
        ValueError: If data contains non-positive values or if sample size is < 2.
      """

      if not all(x > 0 for x in data):
          raise ValueError("Input data must contain only positive values.")

      log_data = np.log(data)
      n = len(log_data)

      if n < 2:
          raise ValueError("Sample size must be at least 2.")

      y_bar = np.mean(log_data) # Sample mean of log-transformed data [cite: 11]
      s_squared = np.var(log_data, ddof=1) # Sample variance of log-transformed data [cite: 11]
      df = n - 1 # Degrees of freedom

      alpha = 1 - confidence_level

      # Calculate the term sqrt(S^2 * (1 + S^2 / 2)) [cite: 27]
      common_term_sqrt = math.sqrt(s_squared * (1 + s_squared / 2.0))

      # Calculate the t-percentile component for the lower limit [cite: 27]
      t_quantile = t.ppf(1 - alpha / 2.0, df)
      lower_term = (t_quantile / math.sqrt(n)) * common_term_sqrt

      # Calculate the q-percentile component for the upper limit [cite: 27]
      chi2_quantile = chi2.ppf(alpha / 2.0, df)
      # Handle potential division by zero or negative value under square root if chi2_quantile is too small or df is small
      if chi2_quantile <= 0 or (n-1)/chi2_quantile <= 1 :
          # In this edge case, the formula for q becomes problematic.
          # Angus notes T(sigma) approaches a limit as sigma -> infinity.
          # However, a practical upper bound might be very large or infinite.
          # Returning infinity or raising an error might be appropriate.
          # For simplicity here, we'll return infinity for the upper bound.
          # A more sophisticated implementation might simulate T(sigma) or use alternative bounds.
           q_quantile = float('inf')
           upper_term = float('inf')
      else:
           q_quantile = math.sqrt((n / 2.0) * (((n - 1) / chi2_quantile) - 1))
           upper_term = (q_quantile / math.sqrt(n)) * common_term_sqrt


      # Calculate the lower and upper limits for ln(theta) [cite: 27]
      ln_theta_lower = y_bar + s_squared / 2.0 - lower_term
      ln_theta_upper = y_bar + s_squared / 2.0 + upper_term

      # Exponentiate to get the confidence interval for theta (the mean)
      theta_lower = math.exp(ln_theta_lower)
      theta_upper = math.exp(ln_theta_upper)

      return (theta_lower, theta_upper)



    @staticmethod
    def lognormal_median_ci(data, confidence_level=0.95):
        """
        Estimates the median and its confidence interval for data assumed
        to be log-normally distributed.

        Args:
            data (array-like): A list, numpy array, or pandas Series of
                               positive numerical data points.
            confidence_level (float): The desired confidence level (e.g., 0.95 for 95%).
                                     Must be between 0 and 1.

        Returns:
            dict: A dictionary containing:
                  'median_estimate': The point estimate of the median.
                  'confidence_interval': A tuple (lower_bound, upper_bound)
                                         for the median.
                  Returns None if input data is invalid (e.g., non-positive values,
                  not enough data points).
        """
        # --- Input Validation ---
        try:
            # Convert to numpy array for easier handling
            data = np.asarray(data)

            # Check for non-positive values (logarithm is undefined)
            if np.any(data <= 0):
                print("Error: Data contains non-positive values. Log-normal "
                      "distribution is only defined for positive values.")
                return None

            # Check for sufficient data
            n = len(data)
            if n < 2:
                print("Error: Need at least two data points to estimate variance.")
                return None

            # Check confidence level validity
            if not (0 < confidence_level < 1):
                print("Error: Confidence level must be between 0 and 1.")
                return None

        except Exception as e:
            print(f"Error processing input data: {e}")
            return None

        # --- Calculations ---
        # 1. Log-transform the data
        log_data = np.log(data)

        # 2. Calculate mean and standard deviation of log-transformed data
        mu_hat = np.mean(log_data)
        sigma_hat = np.std(log_data, ddof=1)  # Use ddof=1 for sample standard deviation

        # 3. Point estimate of the median
        median_estimate = np.exp(mu_hat)

        # 4. Calculate confidence interval for the median
        alpha = 1 - confidence_level
        # Degrees of freedom
        dof = n - 1
        # Critical t-value for two-tailed test
        t_critical = stats.t.ppf(1 - alpha / 2, dof)
        # Standard error of the mean of log-transformed data
        se_mu_hat = sigma_hat / np.sqrt(n)

        # Confidence interval for mu (mean of log-data)
        lower_bound_mu = mu_hat - t_critical * se_mu_hat
        upper_bound_mu = mu_hat + t_critical * se_mu_hat

        # Confidence interval for the median (exponentiate the bounds for mu)
        lower_ci_median = np.exp(lower_bound_mu)
        upper_ci_median = np.exp(upper_bound_mu)

        return  median_estimate, ower_ci_median, upper_ci_median

    @staticmethod
    def bootstrap_mean_ci(data, n_bootstraps=1000, confidence_level=0.95):
        """
        Estimates the mean and confidence interval for a log-normal distribution
        using the bootstrapping method.

        Args:
            data (array-like): A 1D array or list containing the log-normally
                               distributed data.
            n_bootstraps (int): The number of bootstrap samples to generate.
                               Defaults to 1000.
            confidence_level (float): The desired confidence level for the interval.
                                     Must be between 0 and 1. Defaults to 0.95.

        Returns:
            tuple: A tuple containing:
                   - estimated_mean (float): The estimated mean of the log-normal
                                             distribution (mean of bootstrap means).
                   - ci_lower (float): The lower bound of the confidence interval.
                   - ci_upper (float): The upper bound of the confidence interval.
        """
        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1")

        n_samples = len(data)
        if n_samples == 0:
            return np.nan, np.nan, np.nan

        bootstrap_means = []
        for _ in range(n_bootstraps):
            # Resample with replacement
            bootstrap_sample = np.random.choice(data, size=n_samples, replace=True)
            # Calculate the mean of the resample
            bootstrap_means.append(np.mean(bootstrap_sample))

        # Calculate the estimated mean as the mean of the bootstrap means
        estimated_mean = np.mean(bootstrap_means)

        # Calculate the confidence interval using the percentile method
        alpha = 1.0 - confidence_level
        lower_percentile = alpha / 2.0 * 100
        upper_percentile = (1.0 - alpha / 2.0) * 100
        ci_lower, ci_upper = np.percentile(bootstrap_means, [lower_percentile, upper_percentile])

        return estimated_mean, ci_lower, ci_upper

    import numpy as np


    @staticmethod
    def ucl_lands_method_accurate(data: Sequence[float]) -> float:
            """
            Calculate the 95% UCL for a log-normal population mean using Land’s H-statistic
            (Land, 1972).

            Parameters
            ----------
            data
                Array‐like of positive observations.

            Returns
            -------
            float
                Upper 95% confidence limit on the arithmetic mean.
            """
            # --- H-table axes ---
            n_values = np.array([3, 5, 7, 10, 12, 15, 21, 31, 51, 101])
            s_values = np.array([
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0,
                6.0, 7.0, 8.0, 9.0, 10.0
            ])

            H_table = np.array([
                [2.75, 2.035, 1.886, 1.802, 1.775, 1.749, 1.722, 1.701, 1.684, 1.67],
                [3.295, 2.198, 1.992, 1.881, 1.843, 1.809, 1.771, 1.742, 1.718, 1.697],
                [4.109, 2.402, 2.125, 1.977, 1.927, 1.882, 1.833, 1.793, 1.761, 1.733],
                [5.22, 2.631, 2.282, 2.089, 2.026, 1.968, 1.905, 1.856, 1.813, 1.777],
                [6.495, 2.947, 2.465, 2.22, 2.141, 2.068, 1.989, 1.928, 1.876, 1.83],
                [7.807, 3.287, 2.673, 2.368, 2.271, 2.181, 2.085, 2.01, 1.946, 1.891],
                [9.12, 3.662, 2.904, 2.532, 2.414, 2.306, 2.191, 2.102, 2.025, 1.96],
                [10.43, 4.062, 3.155, 2.71, 2.57, 2.443, 2.307, 2.202, 2.112, 2.035],
                [11.74, 4.478, 3.402, 2.902, 2.738, 2.589, 2.432, 2.31, 2.206, 2.117],
                [13.05, 4.905, 3.698, 3.103, 2.915, 2.744, 2.564, 2.423, 2.306, 2.205],
                [16.33, 6.001, 4.426, 3.639, 3.389, 3.163, 2.923, 2.737, 2.58, 2.447],
                [19.6, 7.12, 5.184, 4.207, 3.896, 3.612, 3.311, 3.077, 2.881, 2.713],
                [22.87, 8.25, 5.96, 4.795, 4.422, 4.081, 3.719, 3.437, 3.2, 2.997],
                [26.14, 9.387, 6.747, 5.396, 4.962, 4.564, 4.141, 3.812, 3.533, 3.295],
                [32.69, 11.67, 8.339, 6.621, 6.067, 5.557, 5.013, 4.588, 4.228, 3.92],
                [39.23, 13.97, 9.945, 7.864, 7.191, 6.57, 5.907, 5.388, 4.947, 4.569],
                [45.77, 16.27, 11.56, 9.118, 8.326, 7.596, 6.815, 6.201, 5.681, 5.233],
                [52.31, 18.58, 13.18, 10.38, 9.469, 8.63, 7.731, 7.024, 6.424, 5.908],
                [58.85, 20.88, 14.8, 11.64, 10.62, 9.669, 8.652, 7.854, 7.174, 6.59],
                [65.39, 23.19, 16.43, 12.91, 11.77, 10.71, 9.579, 8.688, 7.929, 7.277],
                [78.47, 27.81, 19.68, 15.45, 14.08, 12.81, 11.44, 10.36, 9.449, 8.661],
                [91.55, 32.43, 22.94, 18.0, 16.39, 14.9, 13.31, 12.05, 10.98, 10.05],
                [104.6, 37.06, 26.2, 20.55, 18.71, 17.01, 15.18, 13.74, 12.51, 11.45],
                [117.7, 41.68, 29.46, 23.1, 21.03, 19.11, 17.05, 15.43, 14.05, 12.85],
                [130.8, 46.31, 32.73, 25.66, 23.35, 21.22, 18.93, 17.13, 15.59, 14.26]
            ])

            # build interpolator (allows extrapolation)
            H_interp = RegularGridInterpolator(
                (s_values, n_values),
                H_table,
                bounds_error=False,
                fill_value=None
            )

            # log-transform data
            log_data = np.log(data)
            n = log_data.size
            mean_log = log_data.mean()
            var_log = log_data.var(ddof=1)
            s = np.sqrt(var_log)

            # lookup H
            H = float(H_interp((s, n)))

            # compute UCL
            ucl = np.exp(
                mean_log
                + 0.5 * var_log
                + H * np.sqrt(var_log / n + var_log**2 / (2 * (n - 1)))
            )
            return ucl


    @staticmethod
    def median(a, axis=None, out=None, overwrite_input=False, keepdims=False):
        """
        Compute the median along the specified axis. Mutuated from numpy.

        Returns the median of the array elements.

        Parameters
        ----------
        a : array_like
            Input array or object that can be converted to an array.
        axis : {int, sequence of int, None}, optional
            Axis or axes along which the medians are computed. The default,
            axis=None, will compute the median along a flattened version of
            the array.
        out : ndarray, optional
            Alternative output array in which to place the result. It must
            have the same shape and buffer length as the expected output,
            but the type (of the output) will be cast if necessary.
        overwrite_input : bool, optional
           If True, then allow use of memory of input array `a` for
           calculations. The input array will be modified by the call to
           `median`. This will save memory when you do not need to preserve
           the contents of the input array. Treat the input as undefined,
           but it will probably be fully or partially sorted. Default is
           False. If `overwrite_input` is ``True`` and `a` is not already an
           `ndarray`, an error will be raised.
        keepdims : bool, optional
            If this is set to True, the axes which are reduced are left
            in the result as dimensions with size one. With this option,
            the result will broadcast correctly against the original `arr`.

        Returns
        -------
        median : ndarray
            A new array holding the result. If the input contains integers
            or floats smaller than ``float64``, then the output data-type is
            ``np.float64``.  Otherwise, the data-type of the output is the
            same as that of the input. If `out` is specified, that array is
            returned instead.
        """
        return np.median(a, axis, out, overwrite_input, keepdims)

    @staticmethod
    def mad(data, axis=None, func=None, ignore_nan=False):
        """
        Calculate the median absolute deviation (MAD) mutuated from astropy.

        The MAD is defined as :math: median(abs(a - median(a))).

        Parameters
        ----------
        data : array-like
            Input array or object that can be converted to an array.
        axis : None, int, or tuple of int, optional
            The axis or axes along which the MADs are computed.  The default
            (`None`) is to compute the MAD of the flattened array.
        func : callable, optional
            The function used to compute the median. Defaults to `numpy.ma.median`
            for masked arrays, otherwise to `numpy.median`.
        ignore_nan : bool
            Ignore NaN values (treat them as if they are not in the array) when
            computing the median.  This will use `numpy.ma.median` if ``axis`` is
            specified, or `numpy.nanmedian` if ``axis==None`` and numpy's version
            is >1.10 because nanmedian is slightly faster in this case.

        Returns
        -------
        mad : float or `~numpy.ndarray`
            The median absolute deviation of the input array.  If ``axis``
            is `None` then a scalar will be returned, otherwise a
            `~numpy.ndarray` will be returned.
        """
        return astropy_median_absolute_deviation(data, axis, func, ignore_nan)

    @staticmethod
    def mad_std(data, axis=None, func=None, ignore_nan=False):
        """
        Calculate a robust standard deviation using the median absolute
        deviation (MAD), mutuated from astropy.

        The standard deviation estimator is given by:

        .. math::

            \sigma \approx \frac{\textrm{MAD}}{\Phi^{-1}(3/4)}
                \approx 1.4826 \ \textrm{MAD}

        where :math: \Phi^{-1}(P) is the normal inverse cumulative
        distribution function evaluated at probability :math: P = 3/4.

        Parameters
        ----------
        data : array-like
            Data array or object that can be converted to an array.
        axis : None, int, or tuple of int, optional
            The axis or axes along which the robust standard deviations are
            computed.  The default (`None`) is to compute the robust
            standard deviation of the flattened array.
        func : callable, optional
            The function used to compute the median. Defaults to `numpy.ma.median`
            for masked arrays, otherwise to `numpy.median`.
        ignore_nan : bool
            Ignore NaN values (treat them as if they are not in the array) when
            computing the median.  This will use `numpy.ma.median` if ``axis`` is
            specified, or `numpy.nanmedian` if ``axis=None`` and numpy's version is
            >1.10 because nanmedian is slightly faster in this case.

        Returns
        -------
        mad_std : float or `~numpy.ndarray`
            The robust standard deviation of the input data.  If ``axis`` is
            `None` then a scalar will be returned, otherwise a
            `~numpy.ndarray` will be returned.
        """
        return astropy_mad_std(data, axis, func, ignore_nan)

    @staticmethod
    def sigma_clip(
        data,
        sigma=3,
        sigma_lower=None,
        sigma_upper=None,
        maxiters=5,
        cenfunc="median",
        stdfunc="std",
        axis=None,
        masked=True,
        return_bounds=False,
        copy=True,
        grow=False,
    ):
        """
        Perform sigma-clipping on the provided data. Mutuated from astropy.

        The data will be iterated over, each time rejecting values that are
        less or more than a specified number of standard deviations from a
        center value.

        Clipped (rejected) pixels are those where:
        
        .. math::

            data < center - (\sigma_{lower} * std)
            data > center + (\sigma_{upper} * std)

        where:

            center = cenfunc(data [, axis=])
            std = stdfunc(data [, axis=])

        Invalid data values (i.e., NaN or inf) are automatically clipped.

        For an object-oriented interface to sigma clipping, see
        :class:`SigmaClip`.

        Parameters
        ----------
        data : array-like or `~numpy.ma.MaskedArray`
            The data to be sigma clipped.

        sigma : float, optional
            The number of standard deviations to use for both the lower
            and upper clipping limit. These limits are overridden by
            ``sigma_lower`` and ``sigma_upper``, if input. The default is 3.

        sigma_lower : float or None, optional
            The number of standard deviations to use as the lower bound for
            the clipping limit. If `None` then the value of ``sigma`` is
            used. The default is `None`.

        sigma_upper : float or None, optional
            The number of standard deviations to use as the upper bound for
            the clipping limit. If `None` then the value of ``sigma`` is
            used. The default is `None`.

        maxiters : int or None, optional
            The maximum number of sigma-clipping iterations to perform or
            `None` to clip until convergence is achieved (i.e., iterate
            until the last iteration clips nothing). If convergence is
            achieved prior to ``maxiters`` iterations, the clipping
            iterations will stop. The default is 5.

        cenfunc : {'median', 'mean'} or callable, optional
            The statistic or callable function/object used to compute
            the center value for the clipping. If using a callable
            function/object and the ``axis`` keyword is used, then it must
            be able to ignore NaNs (e.g., `numpy.nanmean`) and it must have
            an ``axis`` keyword to return an array with axis dimension(s)
            removed. The default is ``'median'``.

        stdfunc : {'std', 'mad_std'} or callable, optional
            The statistic or callable function/object used to compute the
            standard deviation about the center value. If using a callable
            function/object and the ``axis`` keyword is used, then it must
            be able to ignore NaNs (e.g., `numpy.nanstd`) and it must have
            an ``axis`` keyword to return an array with axis dimension(s)
            removed. The default is ``'std'``.

        axis : None or int or tuple of int, optional
            The axis or axes along which to sigma clip the data. If `None`,
            then the flattened data will be used. ``axis`` is passed to the
            ``cenfunc`` and ``stdfunc``. The default is `None`.

        masked : bool, optional
            If `True`, then a `~numpy.ma.MaskedArray` is returned, where
            the mask is `True` for clipped values. If `False`, then a
            `~numpy.ndarray` is returned. The default is `True`.

        return_bounds : bool, optional
            If `True`, then the minimum and maximum clipping bounds are also
            returned.

        copy : bool, optional
            If `True`, then the ``data`` array will be copied. If `False`
            and ``masked=True``, then the returned masked array data will
            contain the same array as the input ``data`` (if ``data`` is a
            `~numpy.ndarray` or `~numpy.ma.MaskedArray`). If `False` and
            ``masked=False``, the input data is modified in-place. The
            default is `True`.

        grow : float or `False`, optional
            Radius within which to mask the neighbouring pixels of those
            that fall outwith the clipping limits (only applied along
            ``axis``, if specified). As an example, for a 2D image a value
            of 1 will mask the nearest pixels in a cross pattern around each
            deviant pixel, while 1.5 will also reject the nearest diagonal
            neighbours and so on.

        Returns
        -------
        result : array-like
            If ``masked=True``, then a `~numpy.ma.MaskedArray` is returned,
            where the mask is `True` for clipped values and where the input
            mask was `True`.

            If ``masked=False``, then a `~numpy.ndarray` is returned.

            If ``return_bounds=True``, then in addition to the masked array
            or array above, the minimum and maximum clipping bounds are
            returned.

            If ``masked=False`` and ``axis=None``, then the output array
            is a flattened 1D `~numpy.ndarray` where the clipped values
            have been removed. If ``return_bounds=True`` then the returned
            minimum and maximum thresholds are scalars.

            If ``masked=False`` and ``axis`` is specified, then the output
            `~numpy.ndarray` will have the same shape as the input ``data``
            and contain ``np.nan`` where values were clipped. If the input
            ``data`` was a masked array, then the output `~numpy.ndarray`
            will also contain ``np.nan`` where the input mask was `True`.
            If ``return_bounds=True`` then the returned minimum and maximum
            clipping thresholds will be `~numpy.ndarray`\\s.
        """
        return astropy_sigma_clip(data, sigma, sigma_lower, sigma_upper, maxiters, cenfunc, stdfunc, axis, masked, return_bounds, copy, grow)

    @staticmethod
    def sigma_clipped_stats(
            data,
            mask=None,
            mask_value=None,
            sigma=3.0,
            sigma_lower=None,
            sigma_upper=None,
            maxiters=5,
            cenfunc="median",
            stdfunc="std",
            std_ddof=0,
            axis=None,
            grow=False,
        ):
        """
        Calculate sigma-clipped statistics on the provided data, mutuated from astropy.

        Parameters
        ----------
        data : array-like or `~numpy.ma.MaskedArray`
            Data array or object that can be converted to an array.

        mask : `numpy.ndarray` (bool), optional
            A boolean mask with the same shape as ``data``, where a `True`
            value indicates the corresponding element of ``data`` is masked.
            Masked pixels are excluded when computing the statistics.

        mask_value : float, optional
            A data value (e.g., ``0.0``) that is ignored when computing the
            statistics. ``mask_value`` will be masked in addition to any
            input ``mask``.

        sigma : float, optional
            The number of standard deviations to use for both the lower
            and upper clipping limit. These limits are overridden by
            ``sigma_lower`` and ``sigma_upper``, if input. The default is 3.

        sigma_lower : float or None, optional
            The number of standard deviations to use as the lower bound for
            the clipping limit. If `None` then the value of ``sigma`` is
            used. The default is `None`.

        sigma_upper : float or None, optional
            The number of standard deviations to use as the upper bound for
            the clipping limit. If `None` then the value of ``sigma`` is
            used. The default is `None`.

        maxiters : int or None, optional
            The maximum number of sigma-clipping iterations to perform or
            `None` to clip until convergence is achieved (i.e., iterate
            until the last iteration clips nothing). If convergence is
            achieved prior to ``maxiters`` iterations, the clipping
            iterations will stop. The default is 5.

        cenfunc : {'median', 'mean'} or callable, optional
            The statistic or callable function/object used to compute
            the center value for the clipping. If using a callable
            function/object and the ``axis`` keyword is used, then it must
            be able to ignore NaNs (e.g., `numpy.nanmean`) and it must have
            an ``axis`` keyword to return an array with axis dimension(s)
            removed. The default is ``'median'``.

        stdfunc : {'std', 'mad_std'} or callable, optional
            The statistic or callable function/object used to compute the
            standard deviation about the center value. If using a callable
            function/object and the ``axis`` keyword is used, then it must
            be able to ignore NaNs (e.g., `numpy.nanstd`) and it must have
            an ``axis`` keyword to return an array with axis dimension(s)
            removed. The default is ``'std'``.

        std_ddof : int, optional
            The delta degrees of freedom for the standard deviation
            calculation. The divisor used in the calculation is ``N -
            std_ddof``, where ``N`` represents the number of elements. The
            default is 0.

        axis : None or int or tuple of int, optional
            The axis or axes along which to sigma clip the data. If `None`,
            then the flattened data will be used. ``axis`` is passed to the
            ``cenfunc`` and ``stdfunc``. The default is `None`.

        grow : float or `False`, optional
            Radius within which to mask the neighbouring pixels of those
            that fall outwith the clipping limits (only applied along
            ``axis``, if specified). As an example, for a 2D image a value
            of 1 will mask the nearest pixels in a cross pattern around each
            deviant pixel, while 1.5 will also reject the nearest diagonal
            neighbours and so on.

        Notes
        -----
        The best performance will typically be obtained by setting
        ``cenfunc`` and ``stdfunc`` to one of the built-in functions
        specified as as string. If one of the options is set to a string
        while the other has a custom callable, you may in some cases see
        better performance if you have the `bottleneck`_ package installed.

        .. _bottleneck:  https://github.com/pydata/bottleneck

        Returns
        -------
        mean, median, stddev : float
            The mean, median, and standard deviation of the sigma-clipped
            data.
        """
        return astropy_sigma_clipped_stats(data, mask, mask_value, sigma, sigma_lower, sigma_upper, maxiters, cenfunc, stdfunc, std_ddof, axis, grow)
   
    @staticmethod
    def biweight_location(data, c=6.0, M=None, axis=None, ignore_nan=False):
        """
        Compute the biweight location.

        The biweight location is a robust statistic for determining the
        central location of a distribution.  It is given by:

        .. math::

            \zeta_{biloc}= M + \frac{\sum_{|u_i|<1}(x_i - M) (1 - u_i^2)^2}{\sum_{|u_i|<1}(1 - u_i^2)^2}

        where :math:`x` is the input data, :math:`M` is the sample median
        (or the input initial location guess) and :math:`u_i` is given by:

        .. math::

            u_{i} = \frac{(x_i - M)}{c \cdot MAD}

        where :math:`c` is the tuning constant and :math:`MAD` is the
        `median absolute deviation
        <https://en.wikipedia.org/wiki/Median_absolute_deviation>`_.  The
        biweight location tuning constant ``c`` is typically 6.0 (the
        default).

        If :math:`MAD` is zero, then the median will be returned.

        Parameters
        ----------
        data : array-like
            Input array or object that can be converted to an array.
            ``data`` can be a `~numpy.ma.MaskedArray`.
        c : float, optional
            Tuning constant for the biweight estimator (default = 6.0).
        M : float or array-like, optional
            Initial guess for the location.  If ``M`` is a scalar value,
            then its value will be used for the entire array (or along each
            ``axis``, if specified).  If ``M`` is an array, then its must be
            an array containing the initial location estimate along each
            ``axis`` of the input array.  If `None` (default), then the
            median of the input array will be used (or along each ``axis``,
            if specified).
        axis : None, int, or tuple of int, optional
            The axis or axes along which the biweight locations are
            computed.  If `None` (default), then the biweight location of
            the flattened input array will be computed.
        ignore_nan : bool, optional
            Whether to ignore NaN values in the input ``data``.

        Returns
        -------
        biweight_location : float or `~numpy.ndarray`
            The biweight location of the input data.  If ``axis`` is `None`
            then a scalar will be returned, otherwise a `~numpy.ndarray`
            will be returned.
        """
        if ignore_nan:
            data = np.ma.masked_invalid(data)
        return astropy_biweight_location(data, c=c, M=M, axis=axis)
    
    @staticmethod
    def biweight_scale(data, c=9.0, M=None, axis=None, modify_sample_size=False, ignore_nan=False):
        """
        Compute the biweight scale.

        The biweight scale is a robust statistic for determining the
        standard deviation of a distribution.  It is the square root of the
        `biweight midvariance.

        It is given by:

        .. math::

            \zeta_{biscl} = \sqrt{n}\frac{\sqrt{\sum_{|u_i| < 1}(x_i - M)^2 (1 - u_i^2)^4}} {|(\sum_{|u_i| < 1}(1 - u_i^2) (1 - 5u_i^2))|}

        where :math:`x` is the input data, :math:`M` is the sample median
        (or the input location) and :math:`u_i` is given by:

        .. math::

            u_{i} = \frac{x_i - M}{c * MAD}

        where :math:`c` is the tuning constant and :math:`MAD` is the
        `median absolute deviation
        <https://en.wikipedia.org/wiki/Median_absolute_deviation>`_.  The
        biweight midvariance tuning constant ``c`` is typically 9.0 (the
        default).

        If :math:`MAD` is zero, then zero will be returned.

        For the standard definition of biweight scale, :math:`n` is the
        total number of points in the array (or along the input ``axis``, if
        specified).  That definition is used if ``modify_sample_size`` is
        `False`, which is the default.

        However, if ``modify_sample_size = True``, then :math:`n` is the
        number of points for which :math:`|u_i| < 1` (i.e. the total number
        of non-rejected values), i.e.

        .. math::

            n = \sum_{|u_i| < 1} 1

        which results in a value closer to the true standard deviation for
        small sample sizes or for a large number of rejected values.

        Parameters
        ----------
        data : array-like
            Input array or object that can be converted to an array.
            ``data`` can be a `~numpy.ma.MaskedArray`.
        c : float, optional
            Tuning constant for the biweight estimator (default = 9.0).
        M : float or array-like, optional
            The location estimate.  If ``M`` is a scalar value, then its
            value will be used for the entire array (or along each ``axis``,
            if specified).  If ``M`` is an array, then its must be an array
            containing the location estimate along each ``axis`` of the
            input array.  If `None` (default), then the median of the input
            array will be used (or along each ``axis``, if specified).
        axis : None, int, or tuple of int, optional
            The axis or axes along which the biweight scales are computed.
            If `None` (default), then the biweight scale of the flattened
            input array will be computed.
        modify_sample_size : bool, optional
            If `False` (default), then the sample size used is the total
            number of elements in the array (or along the input ``axis``, if
            specified), which follows the standard definition of biweight
            scale.  If `True`, then the sample size is reduced to correct
            for any rejected values (i.e. the sample size used includes only
            the non-rejected values), which results in a value closer to the
            true standard deviation for small sample sizes or for a large
            number of rejected values.
        ignore_nan : bool, optional
            Whether to ignore NaN values in the input ``data``.

        Returns
        -------
        biweight_scale : float or `~numpy.ndarray`
            The biweight scale of the input data.  If ``axis`` is `None`
            then a scalar will be returned, otherwise a `~numpy.ndarray`
            will be returned.
        """
        if ignore_nan:
            data = np.ma.masked_invalid(data)
        return astropy_biweight_scale(data, c=c, M=M, axis=axis, modify_sample_size=modify_sample_size)

    @staticmethod
    def trim(a, limits=None, inclusive=(True, True), axis=None):
        """
        Trims an array by masking the data outside some given limits. Mutuated for scipy.

        Returns a masked version of the input array.

        Parameters
        ----------
        a : array_like
            Input array.
        limits : {None, tuple of float}, optional
            Tuple of the percentages to cut on each side of the array, with respect
            to the number of unmasked data, as floats between 0. and 1.
            Noting n the number of unmasked data before trimming, the
            (n*limits[0])th smallest data and the (n*limits[1])th largest data are
            masked, and the total number of unmasked data after trimming
            is n*(1.-sum(limits)). The value of one limit can be set to None to
            indicate an open interval.
        inclusive : {(True, True) tuple}, optional
            Tuple indicating whether the number of data being masked on each side
            should be truncated (True) or rounded (False).
        axis : {None, int}, optional
            Axis along which to trim. If None, the whole array is trimmed, but its
            shape is maintained.
        """
        return scipy_trim(a, limits, inclusive, axis)

    @staticmethod
    def trimmed_mean(a, limits=None, inclusive=(True, True), axis=None):
        """
        Compute the trimmed, mean given a lower and an upper limit. Mutuated from Scipy stats.

        This function finds the arithmetic mean of given values, ignoring values
        outside the given `limits`.

        Parameters
        ----------
        a : array_like
            Array of values.
        limits : None or (lower limit, upper limit), optional
            Values in the input array less than the lower limit or greater than the
            upper limit will be ignored.  When limits is None (default), then all
            values are used.  Either of the limit values in the tuple can also be
            None representing a half-open interval.
        inclusive : (bool, bool), optional
            A tuple consisting of the (lower flag, upper flag).  These flags
            determine whether values exactly equal to the lower or upper limits
            are included.  The default value is (True, True).
        axis : int or None, optional
            Axis along which to compute test. Default is None.

        Returns
        -------
        tmean : ndarray
            Trimmed mean.
        """
        return scipy_tmean(a, limits, inclusive, axis)
    
    @staticmethod
    def trimmed_std(a, limits=(0.1,0.1), inclusive=(1,1), relative=True, axis=None, ddof=0):
        """
        Returns the trimmed standard deviation of the data along the given axis. Mutuated from Scipy stats.

        Parameters
        ----------
        a : array_like
            Input array.
        limits : tuple of float, optional
            The lower and upper fraction of elements to trim. These fractions
            should be between 0 and 1.
        inclusive : tuple of {0, 1}, optional
            Tuple indicating whether the number of data being masked on each side
            should be truncated (1) or rounded (0).
        relative : bool, optional
            Whether to treat the `limits` as relative or absolute positions.
        axis : int, optional
            Axis along which to perform the trimming.
        ddof : int, optional
            Means Delta Degrees of Freedom. The denominator used in the calculations
            is ``n - ddof``, where ``n`` represents the number of elements.
        """
        return scipy_trimmed_std(a, limits, inclusive, relative, axis, ddof)

    @staticmethod
    def trimboth(a, proportiontocut=0.2, axis=0):
        """Slice off a proportion of items from both ends of an array.

        Slice off the passed proportion of items from both ends of the passed
        array (i.e., with `proportiontocut` = 0.1, slices leftmost 10% **and**
        rightmost 10% of scores). The trimmed values are the lowest and
        highest ones.
        Slice off less if proportion results in a non-integer slice index (i.e.
        conservatively slices off `proportiontocut`).

        Parameters
        ----------
        a : array_like
            Data to trim.
        proportiontocut : float
            Proportion (in range 0-1) of total data set to trim of each end.
        axis : int or None, optional
            Axis along which to trim data. Default is 0. If None, compute over
            the whole array `a`.

        Returns
        -------
        out : ndarray
            Trimmed version of array `a`. The order of the trimmed content
            is undefined.

        See Also
        --------
        trim_mean

        Examples
        --------
        Create an array of 10 values and trim 10% of those values from each end:

        >>> import numpy as np
        >>> from scipy import stats
        >>> a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> stats.trimboth(a, 0.1)
        array([1, 3, 2, 4, 5, 6, 7, 8])

        Note that the elements of the input array are trimmed by value, but the
        output array is not necessarily sorted.

        The proportion to trim is rounded down to the nearest integer. For
        instance, trimming 25% of the values from each end of an array of 10
        values will return an array of 6 values:

        >>> b = np.arange(10)
        >>> stats.trimboth(b, 1/4).shape
        (6,)

        Multidimensional arrays can be trimmed along any axis or across the entire
        array:

        >>> c = [2, 4, 6, 8, 0, 1, 3, 5, 7, 9]
        >>> d = np.array([a, b, c])
        >>> stats.trimboth(d, 0.4, axis=0).shape
        (1, 10)
        >>> stats.trimboth(d, 0.4, axis=1).shape
        (3, 2)
        >>> stats.trimboth(d, 0.4, axis=None).shape
        (6,)

        """
        return scipy_trimboth(a, proportiontocut, axis)

    @staticmethod
    def trimtail(data, proportiontocut=0.2, tail='left', inclusive=(True, True), axis=None):
        """
        Trims the data by masking values from one tail.

        Parameters
        ----------
        data : array_like
            Data to trim.
        proportiontocut : float, optional
            Percentage of trimming. If n is the number of unmasked values
            before trimming, the number of values after trimming is
            ``(1 - proportiontocut) * n``.  Default is 0.2.
        tail : {'left', 'right'}, optional
            If 'left' the `proportiontocut` lowest values will be masked.
            If 'right' the `proportiontocut` highest values will be masked.
            Default is 'left'.
        inclusive : {(bool, bool) tuple}, optional
            Tuple indicating whether the number of data being masked on each side
            should be rounded (True) or truncated (False).  Default is
            (True, True).
        axis : int, optional
            Axis along which to perform the trimming.
            If None, the input array is first flattened.  Default is None.

        Returns
        -------
        trimtail : ndarray
            Returned array of same shape as `data` with masked tail values.
        """
        return scipy_trimtail(data, proportiontocut, tail, inclusive, axis)

    @staticmethod
    def winsorize(a, limits=None, inclusive=(True, True), inplace=False, axis=None, nan_policy='propagate'):
        """
        Returns a Winsorized version of the input array. Mutuated from Scipy

        The (limits[0])th lowest values are set to the (limits[0])th percentile,
        and the (limits[1])th highest values are set to the (1 - limits[1])th
        percentile.
        Masked values are skipped.

        Parameters
        ----------
        a : sequence
            Input array.
        limits : {None, tuple of float}, optional
            Tuple of the percentages to cut on each side of the array, with respect
            to the number of unmasked data, as floats between 0. and 1.
            Noting n the number of unmasked data before trimming, the
            (n*limits[0])th smallest data and the (n*limits[1])th largest data are
            masked, and the total number of unmasked data after trimming
            is n*(1.-sum(limits)) The value of one limit can be set to None to
            indicate an open interval.
        inclusive : {(True, True) tuple}, optional
            Tuple indicating whether the number of data being masked on each side
            should be truncated (True) or rounded (False).
        inplace : {False, True}, optional
            Whether to winsorize in place (True) or to use a copy (False)
        axis : {None, int}, optional
            Axis along which to trim. If None, the whole array is trimmed, but its
            shape is maintained.
        nan_policy : {'propagate', 'raise', 'omit'}, optional
            Defines how to handle when input contains nan.
            The following options are available (default is 'propagate'):

              * 'propagate': allows nan values and may overwrite or propagate them
              * 'raise': throws an error
              * 'omit': performs the calculations ignoring nan values

        Notes
        -----
        This function is applied to reduce the effect of possibly spurious outliers
        by limiting the extreme values.

        Returns
        -------
        winsorized : ndarray
            Winsorized array.
        """
        return scipy_winsorize(a, limits, inclusive, inplace, axis, nan_policy)

    @staticmethod
    def winsorized_mean(a, limits=None, inclusive=(True, True), inplace=False, axis=None, nan_policy='propagate'):
        """
        Returns a Winsorized mean of the input array.

        The (limits[0])th lowest values are set to the (limits[0])th percentile,
        and the (limits[1])th highest values are set to the (1 - limits[1])th
        percentile.
        Masked values are skipped.

        Parameters
        ----------
        a : sequence
            Input array.
        limits : {None, tuple of float}, optional
            Tuple of the percentages to cut on each side of the array, with respect
            to the number of unmasked data, as floats between 0. and 1.
            Noting n the number of unmasked data before trimming, the
            (n*limits[0])th smallest data and the (n*limits[1])th largest data are
            masked, and the total number of unmasked data after trimming
            is n*(1.-sum(limits)) The value of one limit can be set to None to
            indicate an open interval.
        inclusive : {(True, True) tuple}, optional
            Tuple indicating whether the number of data being masked on each side
            should be truncated (True) or rounded (False).
        inplace : {False, True}, optional
            Whether to winsorize in place (True) or to use a copy (False)
        axis : {None, int}, optional
            Axis along which to trim. If None, the whole array is trimmed, but its
            shape is maintained.
        nan_policy : {'propagate', 'raise', 'omit'}, optional
            Defines how to handle when input contains nan.
            The following options are available (default is 'propagate'):

              * 'propagate': allows nan values and may overwrite or propagate them
              * 'raise': throws an error
              * 'omit': performs the calculations ignoring nan values

        Notes
        -----
        This function is applied to reduce the effect of possibly spurious outliers
        by limiting the extreme values.

        Returns
        -------
        winsorized_mean : float
            Winsorized mean of the array.
        """
        return np.mean(scipy_winsorize(a, limits, inclusive, inplace, axis, nan_policy))

    @staticmethod
    def winsorized_std(a, ddof=1, limits=None, inclusive=(True, True), inplace=False, axis=None, nan_policy='propagate'):
        """
        Returns a Winsorized Standard Deviation of the input array.

        The (limits[0])th lowest values are set to the (limits[0])th percentile,
        and the (limits[1])th highest values are set to the (1 - limits[1])th
        percentile.
        Masked values are skipped.

        Parameters
        ----------
        a : sequence
            Input array.
        ddof : int, optional
            Delta Degrees of Freedom. The denominator used in calculations is `N - ddof`, 
            where `N` represents the number of elements. By default ddof is one.
        limits : {None, tuple of float}, optional
            Tuple of the percentages to cut on each side of the array, with respect
            to the number of unmasked data, as floats between 0. and 1.
            Noting n the number of unmasked data before trimming, the
            (n*limits[0])th smallest data and the (n*limits[1])th largest data are
            masked, and the total number of unmasked data after trimming
            is n*(1.-sum(limits)) The value of one limit can be set to None to
            indicate an open interval.
        inclusive : {(True, True) tuple}, optional
            Tuple indicating whether the number of data being masked on each side
            should be truncated (True) or rounded (False).
        inplace : {False, True}, optional
            Whether to winsorize in place (True) or to use a copy (False).
        axis : {None, int}, optional
            Axis along which to trim. If None, the whole array is trimmed, but its
            shape is maintained.
        nan_policy : {'propagate', 'raise', 'omit'}, optional
            Defines how to handle when input contains nan.
            The following options are available (default is 'propagate'):

              * 'propagate': allows nan values and may overwrite or propagate them
              * 'raise': throws an error
              * 'omit': performs the calculations ignoring nan values

        Returns
        -------
        winsorized_std : float
            Winsorized standard deviation of the array.
        """
        return np.std(scipy_winsorize(a, limits, inclusive, inplace, axis, nan_policy), ddof=ddof)

    @staticmethod
    def Huber(data, c=1.5, tol=1e-08, maxiter=30, norm=None):
        """
        Huber's proposal 2 for estimating location and scale jointly. 
        Return joint estimates of Huber's scale and location. 
        Mutuated from statsmodels.robust.

        Parameters
        ----------
        c : float, optional
            Threshold used in threshold for :math: \chi=\psi^2.  Default value is 1.5.
        tol : float, optional
            Tolerance for convergence.  Default value is 1e-08.
        maxiter : int, optional
            Maximum number of iterations.  Default value is 30.
        norm : statsmodels.robust.norms.RobustNorm, optional
            A robust norm used in M estimator of location. If None,
            the location estimator defaults to a one-step
            fixed point version of the M-estimator using Huber's T.

        Returns
        -------
        huber : tuple
            Returns a tuple (location, scale) of the joint estimates.
        """
        huber_proposal_2 = sm.robust.Huber(c, tol, maxiter, norm)
        return huber_proposal_2(data)



