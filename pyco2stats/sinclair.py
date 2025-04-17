import numpy as np
from scipy.stats import norm


class Sinclair:
    @staticmethod
    def get_raw_data(my_data):
        """
        Compute the theoretical quantiles and sorted data values for a QQ plot.

        Parameters:
        - my_data (array-like): The input data for which QQ plot data is to be calculated.
        
        Returns:
        - osm (ndarray): Theoretical quantiles from the normal distribution.
        - osr (ndarray): Ordered (sorted) values of the input data.
        """
        # Calculate uniform order statistic medians
        osm_uniform = Sinclair._calc_uniform_order_statistic_medians(len(my_data))
        
        # Convert uniform medians to quantiles of the standard normal distribution
        osm = norm.ppf(osm_uniform)
        
        # Sort the input data
        osr = np.sort(my_data)
        
        return osm, osr

    @staticmethod
    def calculate_combined_population(meds, stds, fds, mminy, mmaxy, n=100):
        """
        Calculate the combined Gaussian mixture distribution and map it to sigma values.

        This function generates transformed sigma values for a range of ylon values,
        considering a mixture of Gaussian distributions with specified means, standard deviations, and weights.
        
        Parameters:
        
        meds (list or array): List or array of means for each Gaussian component.
        stds (list or array): List or array of standard deviations for each Gaussian component.
        fds (list or array): List or array of weights (relative importance) for each Gaussian component.
        mminy (int or float): Minimum value for generating ylon values.
        mmaxy (int or float): Maximum value for generating ylon values.
        n (int): Number of points to generate between mminy and mmaxy.
        
        Returns:
        ixe (numpy array): Array of transformed sigma values corresponding to each ylon value.
        ylon_1 (numpy array): Array of ylon values, linearly spaced between mminy and mmaxy.
        """
        # Generate n evenly spaced ylon values from mminy to mmaxy
        ylon_1 = np.linspace(mminy, mmaxy, n)
        
        # Initialize the array to hold transformed sigma values
        ixe = np.zeros_like(ylon_1)

        for index, ylon in enumerate(ylon_1):
            combined_prob = 0
            for med, std, fd in zip(meds, stds, fds):
                # Calculate the standardized value (z-score)
                z = (ylon - med) / std
                
                # Use the CDF to get the probability and adjust by weight
                prob = norm.cdf(z)  # Get the CDF value for the z-score
                
                combined_prob += fd * prob  # Combine the probabilities using the weights
            
            # Normalize the combined probability
            combined_prob /= sum(fds)
            
            # Convert the combined probability to sigma values using the inverse CDF
            if combined_prob >= 0.5:
                sigma_value = norm.ppf(combined_prob)  # Positive side
            else:
                sigma_value = -norm.ppf(1 - combined_prob)  # Negative side
            
            # Store the sigma value in the ixe array
            ixe[index] = sigma_value
        
        return ixe, ylon_1

    @staticmethod
    def _calc_uniform_order_statistic_medians(n):
        """
        Calculate the uniform order statistic medians for sample size n.
        
        This is used to estimate the theoretical quantiles for the QQ plot.
        
        Parameters:
        n (int) : The number of data points.
        
        Returns:
        medians (ndarray) : The medians of the uniform order statistics.
        """
        v = np.empty(n, dtype=np.float64)
        v[-1] = 0.5 ** (1.0 / n)  # Calculate the median for the last order statistic
        v[0] = 1 - v[-1]  # Calculate the median for the first order statistic
        i = np.arange(2, n)  # Indices for the remaining order statistics
        v[1:-1] = (i - 0.3175) / (n + 0.365)  # Compute the medians for the intermediate order statistics
        return v
