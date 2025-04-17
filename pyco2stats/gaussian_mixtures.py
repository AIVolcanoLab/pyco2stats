import numpy as np
from scipy.stats import norm
import torch
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
import warnings

class GMM:
    @staticmethod
    def gaussian_mixture_em(data, n_components, max_iter=100, tol=1e-6):
        """
        Fit a Gaussian Mixture Model (GMM) to the given data using the Expectation-Maximization (EM) algorithm.
        
        From 10.1016/j.ijggc.2016.02.012

        Parameters:

        -----------------
        
        data (array-like): The input data to fit the GMM to.
        n_components (int): The number of Gaussian components in the mixture.
        max_iter (int): The maximum number of iterations for the EM algorithm. Default is 100.
        tol (float): The tolerance for convergence. Default is 1e-6.

        Returns:

        -----------------
        
        means (ndarray): The means of the Gaussian components.
        std_devs (ndarray): The standard deviations of the Gaussian components.
        weights (ndarray): The weights (mixing proportions) of the Gaussian components.
        log_likelihoods (list): The log-likelihood values over the iterations.
        """
        
        n = len(data)  # Number of data points

        # Randomly initialize the parameters for the Gaussian components
        np.random.seed(42)  # Seed for reproducibility
        means = np.random.choice(data, n_components)  # Randomly pick initial means from the data
        std_devs = np.random.random(n_components)  # Initialize standard deviations with random values
        weights = np.ones(n_components) / n_components  # Initialize weights uniformly

        log_likelihoods = []  # List to store the log-likelihood values over the iterations
        
        for iteration in range(max_iter):
            # E-step: Compute the responsibilities (posterior probabilities) for each data point and component
            responsibilities = np.zeros((n, n_components))  # Initialize the responsibilities matrix
            for k in range(n_components):
                # Calculate the responsibility of component k for each data point
                responsibilities[:, k] = weights[k] * norm.pdf(data, means[k], std_devs[k])
            
            # Normalize responsibilities so that they sum to 1 for each data point
            sum_responsibilities = responsibilities.sum(axis=1, keepdims=True)
            responsibilities /= sum_responsibilities
            
            # M-step: Update the parameters (means, standard deviations, and weights) based on the responsibilities
            N_k = responsibilities.sum(axis=0)  # Effective number of points assigned to each component
            weights = N_k / n  # Update weights as the fraction of total points assigned to each component
            
            # Update means as the weighted average of the data points
            means = (responsibilities.T @ data) / N_k
            
            # Update standard deviations as the weighted standard deviation of the data points
            std_devs = np.sqrt(np.sum(responsibilities * (data[:, np.newaxis] - means)**2, axis=0) / N_k)
            
            # Compute the log-likelihood of the current parameter estimates
            log_likelihood = np.sum(np.log(sum_responsibilities))
            log_likelihoods.append(log_likelihood)
            
            # Check for convergence: if the log-likelihood improvement is below the tolerance, stop the algorithm
            if iteration > 0 and np.abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
                break
        
        # Return the optimized parameters and the log-likelihood history
        return means, std_devs, weights, log_likelihoods

    @staticmethod
    def gaussian_mixture_sklearn(X, n_components = 3, max_iter = 10, tol = 1e-10, n_init = 20, suppress_warnings= True, covariance_type = 'spherical'  ):
        """
        Fit a Gaussian Mixture Model (GMM) mutuated from sklearn.

        Parameters:
        - X (array-like): The input data to fit the GMM to.
        - n_components (int): The number of Gaussian components in the mixture.
        - max_iter (int): The maximum number of iterations for the EM algorithm. Default is 100.
        - tol (float): The tolerance for convergence. Default is 1e-10.
        - n_init (int): The number of initializations to perform. The best results are kept. Default is 20.
        - suppress_warnings (bool): If True, suppresses the generation of warnings. Default is True.
        - covariance_type (string): can be 'full', 'tied', 'diag' or 'spherical'. Describes the type of covariance parameters to use. Default is 'spherical'.

        Returns:
        - means (ndarray): The means of the Gaussian components.
        - std_devs (ndarray): The standard deviations of the Gaussian components.
        - weights (ndarray): The weights (mixing proportions) of the Gaussian components.
        - max_iter (int): maximum number of iteration (given as input).
        - log_likelihoods (list): The log-likelihood values over the iterations.
        """
        
        # Standardize data to avoid numerical issues
        scaler = StandardScaler()
        X_scaled =  scaler.fit_transform(X)

        # Suppress ConvergenceWarning for clean output
        if suppress_warnings == True:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=ConvergenceWarning)
              
                # Fit GMM with the parameters
                gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=42, 
                                      max_iter=max_iter, tol=tol, n_init=n_init, init_params='kmeans')
                gmm.fit(X_scaled)
        else:
            # Fit GMM with the parameters
                gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=42, 
                                      max_iter=max_iter, tol=tol, n_init=n_init, init_params='kmeans')
                gmm.fit(X_scaled)


        # Get the optimized parameters
        means = gmm.means_.flatten()
        std_devs = np.sqrt(gmm.covariances_.flatten()) if covariance_type == 'full' else np.sqrt(gmm.covariances_)
        weights = gmm.weights_

        # Inverse transform the means to the original scale
        original_means =  scaler.inverse_transform(gmm.means_).flatten()
        original_std_devs = std_devs  * scaler.scale_

        # Custom tracking of log-likelihood over iterations
        log_likelihoods = []
        # For tracking, we simulate iterations manually
        for i in range(1, max_iter + 1):
            if suppress_warnings == True:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=ConvergenceWarning)
                    gmm_iter = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=42, 
                                               max_iter=i, tol=tol, n_init=n_init, init_params='kmeans')
                    gmm_iter.fit(X_scaled)
                    log_likelihoods.append(gmm_iter.lower_bound_)
            else:
                gmm_iter = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=42, 
                                               max_iter=i, tol=tol, n_init=n_init, init_params='kmeans')
                gmm_iter.fit(X_scaled)
                log_likelihoods.append(gmm_iter.lower_bound_)

        return original_means, original_std_devs, weights, max_iter, log_likelihoods

    @staticmethod
    def constrained_gaussian_mixture(X, mean_constraints, std_constraints, n_components, n_epochs=5000, lr=0.001, verbose=True):
        """
        Optimize a Gaussian Mixture Model (GMM) using PyTorch with specified constraints on means and standard deviations.
        
        Parameters:
        - X (array-like): Input data to fit the GMM.
        - mean_constraints (list of tuples): List of tuples specifying (min, max) constraints for each component's mean.
        - std_constraints (list of tuples): List of tuples specifying (min, max) constraints for each component's standard deviation.
        - n_components (int): Number of Gaussian components in the mixture.
        - n_epochs (int): Number of iterations for optimization. Default is 5000.
        - lr (float): Learning rate for the optimizer. Default is 0.001.
        - verbose (bool): If True, prints progress every 200 epochs. Default is True.
        
        Returns:
        - optimized_means (ndarray): Optimized means of the Gaussian components.
        - optimized_stds (ndarray): Optimized standard deviations of the Gaussian components.
        - optimized_weights (ndarray): Optimized weights (mixing proportions) of the Gaussian components.
        """
        # Convert input data to a PyTorch tensor
        X = torch.tensor(X, dtype=torch.float32)

        # Initialize the means, standard deviations, and weights with initial values
        # Initialize the means by sampling within the mean constraints
        initial_means = torch.tensor([np.random.uniform(low=mean_constraints[i][0], high=mean_constraints[i][1])
                                      for i in range(n_components)], requires_grad=True)

        # Initialize the standard deviations by sampling within the std constraints
        initial_stds = torch.tensor([np.random.uniform(low=std_constraints[i][0], high=std_constraints[i][1])
                                     for i in range(n_components)], requires_grad=True)

        # Initialize weights uniformly
        initial_weights = torch.tensor([1 / n_components for _ in range(n_components)], requires_grad=True)

        # Define the optimizer
        optimizer = torch.optim.Adam([initial_means, initial_stds, initial_weights], lr=lr)

        def apply_constraints(means, stds, weights):
            """
            Apply constraints to ensure that the means and standard deviations stay within specified bounds,
            and that weights are positive and normalized.
            """
            with torch.no_grad():
                for i in range(n_components):
                    # Clamp means and standard deviations to their respective constraints
                    means[i].clamp_(mean_constraints[i][0], mean_constraints[i][1])
                    stds[i].clamp_(std_constraints[i][0], std_constraints[i][1])
                # Ensure weights are positive and normalize them to sum to 1
                weights.clamp_(min=0)
                weights /= weights.sum()

        # Training loop
        for epoch in range(n_epochs):
            optimizer.zero_grad()

            # Compute the negative log-likelihood
            log_likelihood = 0
            for x in X:
                mixture_prob = 0
                for j in range(n_components):
                    weight = initial_weights[j]
                    mean = initial_means[j]
                    std = initial_stds[j]
                    # Compute the probability density function (PDF) for the Gaussian component
                    mixture_prob += weight * torch.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
                log_likelihood += torch.log(mixture_prob + 1e-10)  # Add a small value to prevent log(0)

            # Normalize the negative log-likelihood
            loss = -log_likelihood.mean()

            # Backpropagation and optimization step
            loss.backward()
            optimizer.step()

            # Apply constraints to the parameters
            apply_constraints(initial_means, initial_stds, initial_weights)

            # Print progress every 200 epochs if verbose is True
            if verbose and epoch % 200 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item()}')

        # Extract the optimized parameters
        optimized_means = initial_means.detach().numpy()
        optimized_stds = initial_stds.detach().numpy()
        optimized_weights = initial_weights.detach().numpy()

        return optimized_means, optimized_stds, optimized_weights

    @staticmethod
    def gaussian_mixture_pdf(x, meds, stds, weights):
        """
        Compute the PDF of a Gaussian Mixture Model.
        
        Parameters:
        - x (array-like): x values at which to compute the PDF.
        - means (list or array): means for each Gaussian component.
        - stds (list or array): standard deviations for each Gaussian component.
        - weights (list or array): weights (relative importance that must sum to 1) for each Gaussian component.
        
        Returns:
        - pdf (array): The computed PDF values for the Gaussian Mixture Model at each x.
        """
        
        
        # Initialize the PDF to zero
        pdf = np.zeros_like(x, dtype=float)
        
        # Compute the PDF for each Gaussian component and sum them up
        for med, std, weight in zip(meds, stds, weights):
            pdf += weight * norm.pdf(x, med, std)
        
        return pdf

