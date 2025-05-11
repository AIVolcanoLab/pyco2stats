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
    def gaussian_mixture_em(data, n_components, max_iter=10000, tol=1e-7, random_state=None):
        """
        Fit a Gaussian Mixture Model (GMM) to the given data using the Expectation-Maximization (EM) algorithm.

        From 10.1016/j.ijggc.2016.02.012

        Parameters
        ----------
        data : array
            The input data to fit the GMM to.
        n_components : int
            The number of Gaussian components in the mixture.
        max_iter : int
            The maximum number of iterations for the EM algorithm. Default is 100.
        tol : float 
            The tolerance for convergence. Default is 1e-6.
        random_state : int, Generator, or None 
            Controls the randomness for initialization.
            If int, uses it as a seed. If Generator, uses it directly.
            If None, uses the global random state (non-deterministic).

        Returns
        -------
        means : array
            The means of the Gaussian components.
        std_devs : array
            The standard deviations of the Gaussian components.
        weights : array
            The weights (mixing proportions) of the Gaussian components.
        log_likelihoods : list
            The log-likelihood values over the iterations.
        """

        # Explicitly ensure data is a numpy array at the start
        data = np.asarray(data)

        n = len(data)  # Number of data points

        # Create a Generator instance based on random_state
        # This is the recommended way to handle randomness
        rng = np.random.default_rng(random_state)


        # Randomly initialize the parameters for the Gaussian components
        # Use the generator instance `rng` for all random operations within this function

        # Initialize means by sampling from the data using the generator
        # Added check for n >= n_components for replace=False
        means = rng.choice(data.flatten() if data.ndim > 1 else data, n_components, replace=False) if n >= n_components else rng.choice(data.flatten() if data.ndim > 1 else data, n_components, replace=True)

        # Initialize standard deviations using the generator
        std_devs = rng.random(n_components) * np.std(data) * 0.1 + 1e-6 # Small random positive stds

        weights = np.ones(n_components) / n_components  # Initialize weights uniformly

        log_likelihoods = []

        # --- EM Algorithm Loop ---
        # Ensure data is 1D for norm.pdf if needed, or handle reshaping inside loop if necessary
        # The original code seems to assume 1D data for norm.pdf, keep this consistent.
        # If data needs to be 2D for matrix operations later, reshape as needed *after* initialization.
        # Based on the provided snippet, data seems to remain 1D and relies on broadcasting or ufuncs.

        for iteration in range(max_iter):
            # E-step: Compute the responsibilities (posterior probabilities) for each data point and component
            responsibilities = np.zeros((n, n_components))
            for k in range(n_components):
                # Ensure std_devs[k] is positive for norm.pdf
                std = std_devs[k] if std_devs[k] > 0 else 1e-9
                # Assuming norm.pdf handles 1D data correctly
                responsibilities[:, k] = weights[k] * norm.pdf(data, means[k], std)


            # Normalize responsibilities
            sum_responsibilities = responsibilities.sum(axis=1, keepdims=True)
            sum_responsibilities = np.where(sum_responsibilities == 0, 1e-9, sum_responsibilities)
            responsibilities /= sum_responsibilities


            # M-step: Update parameters
            N_k = responsibilities.sum(axis=0)
            N_k_safe = np.where(N_k == 0, 1e-9, N_k)

            weights = N_k / n
            weights /= np.sum(weights) # Re-normalize


            # Update means
            # Assuming data is 1D and responsibilities is (n, n_components)
            means = (responsibilities.T @ data) / N_k_safe

            # Update standard deviations
            variance = np.sum(responsibilities * (data[:, np.newaxis] - means)**2, axis=0) / N_k_safe
            std_devs = np.sqrt(np.maximum(variance, 0)) # Ensure argument to sqrt is non-negative


            # Compute log-likelihood
            weighted_pdfs = np.zeros((n, n_components))
            for k in range(n_components):
                 std = std_devs[k] if std_devs[k] > 0 else 1e-9
                 weighted_pdfs[:, k] = weights[k] * norm.pdf(data, means[k], std)

            total_pdf = np.sum(weighted_pdfs, axis=1)
            log_likelihood = np.sum(np.log(total_pdf + 1e-9)) # Add epsilon before log

            log_likelihoods.append(log_likelihood)

            # Check for convergence
            if iteration > 0 and np.abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
                break

        # --- End of EM Algorithm Loop ---


        # Return the optimized parameters and the log-likelihood history
        return means, std_devs, weights, log_likelihoods

    @staticmethod
    def gaussian_mixture_sklearn(X, n_components = 3, max_iter = 10, tol = 1e-10, n_init = 20, suppress_warnings= True, covariance_type = 'spherical'  ):
        """
        Fit a Gaussian Mixture Model (GMM) mutuated from sklearn.

        Parameters
        ----------
        X : array
            The input data to fit the GMM to.
        n_components : int 
            The number of Gaussian components in the mixture.
        max_iter : int
            The maximum number of iterations for the EM algorithm. Default is 100.
        tol : float 
            The tolerance for convergence. Default is 1e-10.
        n_init : int
            The number of initializations to perform. The best results are kept. Default is 20.
        suppress_warnings : bool 
            If True, suppresses the generation of warnings. Default is True.
        covariance_type : string
            Can be 'full', 'tied', 'diag' or 'spherical'. Describes the type of covariance parameters to use. Default is 'spherical'.

        Returns
        -------
        means : array
            The means of the Gaussian components.
        std_devs : array
            The standard deviations of the Gaussian components.
        weights : array
            The weights (mixing proportions) of the Gaussian components.
        max_iter : int
            Maximum number of iteration (given as input).
        log_likelihoods : list 
            The log-likelihood values over the iterations.
        """
        
        X = X.reshape(-1, 1)

        # Standardize data to avoid numerical issues
        scaler = StandardScaler()
        X_scaled =  scaler.fit_transform(X)

        # Suppress ConvergenceWarning for clean output
        if suppress_warnings == True:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=ConvergenceWarning)
              
                # Fit GMM with the parameters
                gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type,  
                                      max_iter=max_iter, tol=tol, n_init=n_init, random_state=42, init_params='random_from_data')
                gmm.fit(X_scaled)
        else:
            # Fit GMM with the parameters
                gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type,  
                                      max_iter=max_iter, tol=tol, n_init=n_init, random_state=42, init_params='random_from_data')
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
                    gmm_iter = GaussianMixture(n_components=n_components, covariance_type=covariance_type,  
                                               max_iter=i, tol=tol, n_init=n_init, init_params='random_from_data')
                    gmm_iter.fit(X_scaled)
                    log_likelihoods.append(gmm_iter.lower_bound_)
            else:
                gmm_iter = GaussianMixture(n_components=n_components, covariance_type=covariance_type,  
                                               max_iter=i, tol=tol, n_init=n_init, init_params='random_from_data')
                gmm_iter.fit(X_scaled)
                log_likelihoods.append(gmm_iter.lower_bound_)

        return original_means, original_std_devs, weights,  log_likelihoods



    @staticmethod
    def constrained_gaussian_mixture(X, mean_constraints, std_constraints, n_components, n_epochs=5000,
                                              lr=0.001, verbose=True):
        """
        Optimize a Gaussian Mixture Model (GMM) using PyTorch with specified constraints on means and standard deviations.
        Uses Softmax for stable weight optimization and LogSumExp for numerical stability.

        Parameters
        ----------
        X : array
            Input data to fit the GMM. Should be 1D for this implementation (univariate GMM).
        mean_constraints : list of tuples
            List of tuples specifying (min, max) constraints for each component's mean. Length must equal n_components.
        std_constraints : list of tuples
            List of tuples specifying (min, max) constraints for each component's standard deviation. 
            Lower bound should be > 0 for numerical stability. Length must equal n_components.
        n_components : int
            Number of Gaussian components in the mixture.
        n_epochs : int
            Number of iterations for optimization. Default is 5000.
        lr : float
            Learning rate for the optimizer. Default is 0.001.
        verbose : bool
            If True, prints progress every 200 epochs. Default is True.

        Returns
        -------
        optimized_means : array
            Optimized means of the Gaussian components.
        optimized_stds : array
            Optimized standard deviations of the Gaussian components.
        optimized_weights : array
            Optimized weights (mixing proportions) of the Gaussian components.
        """
        if len(mean_constraints) != n_components or len(std_constraints) != n_components:
            raise ValueError("Length of constraints lists must match n_components.")

        # Convert input data to a PyTorch tensor
        X = torch.tensor(X, dtype=torch.float32)

        # Initialize the parameters
        # Initialize means by sampling within the mean constraints
        initial_means = torch.tensor([np.random.uniform(low=mean_constraints[i][0], high=mean_constraints[i][1])
                                      for i in range(n_components)], requires_grad=True)

        # Initialize standard deviations by sampling within the std constraints
        # Ensure initial stds are within valid positive range
        initial_stds = torch.tensor(
            [np.random.uniform(low=max(std_constraints[i][0], 1e-6), high=std_constraints[i][1])  # Ensure positive init
             for i in range(n_components)], requires_grad=True)

        # Initialize logits for weights (softmax will be applied)
        # Initializing logits to zeros results in uniform initial weights
        initial_logits = torch.zeros(n_components, requires_grad=True)

        # Define the optimizer - optimizing means, stds, and logits
        optimizer = torch.optim.Adam([initial_means, initial_stds, initial_logits], lr=lr)

        def apply_constraints_stds_means(means, stds):
            """
            Apply constraints to ensure means and standard deviations stay within specified bounds.
            This is done outside the gradient flow using torch.no_grad().
            """
            with torch.no_grad():
                for i in range(n_components):
                    # Clamp means
                    means[i].clamp_(mean_constraints[i][0], mean_constraints[i][1])
                    # Clamp standard deviations - ensure lower bound is positive
                    stds[i].clamp_(max(std_constraints[i][0], 1e-6), std_constraints[i][1])

        # Training loop
        for epoch in range(n_epochs):
            optimizer.zero_grad()

            # Get current weights by applying softmax to logits
            weights = torch.softmax(initial_logits, dim=0)

            # Calculate the log-likelihood of the data given the current GMM parameters
            # Use vectorized operations and log-sum-exp for stability

            # Calculate log probabilities for each data point under each Gaussian component
            log_probs_components = torch.zeros(len(X), n_components)
            for j in range(n_components):
                # Use torch.distributions for more numerical stability in log_prob
                # Ensure std is positive, add a small epsilon if it could become zero (though clamping should help)
                std_j = initial_stds[j]  # + 1e-6 # Add epsilon if lower bound constraint is 0 or could be reached
                try:
                    dist = torch.distributions.Normal(initial_means[j], std_j)
                    log_probs_components[:, j] = dist.log_prob(X)
                except ValueError as e:
                    print(
                        f"Epoch {epoch}, Component {j}: Error with parameters mean={initial_means[j].item()}, std={std_j.item()}")
                    raise e

            # Calculate the log of the weighted probabilities: log(weight) + log(prob)
            # Add a small epsilon to weights before taking log if weights could become zero
            # (softmax ensures positivity, but very small values are possible)
            log_weighted_probs = log_probs_components + torch.log(weights + 1e-10)

            # Use log-sum-exp to get the log-likelihood for each data point: log(sum(weight * prob))
            log_likelihood_per_sample = torch.logsumexp(log_weighted_probs, dim=1)

            # The total log-likelihood is the sum of log-likelihoods for each data point
            # The loss is the negative mean log-likelihood
            loss = -log_likelihood_per_sample.mean()

            # Backpropagation and optimization step
            loss.backward()
            optimizer.step()

            # Apply constraints to means and standard deviations (weights are handled by softmax)
            apply_constraints_stds_means(initial_means, initial_stds)

            # Print progress every 200 epochs if verbose is True
            if verbose and epoch % 200 == 0:
                # Calculate current weights for printing
                current_weights = torch.softmax(initial_logits, dim=0).detach().numpy()
                current_means = initial_means.detach().numpy()
                current_stds = initial_stds.detach().numpy()
                print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
                # Optionally print current parameters to see how they evolve
                # print(f'  Weights: {current_weights}')
                # print(f'  Means: {current_means}')
                # print(f'  Stds: {current_stds}')

        # Extract the optimized parameters
        optimized_means = initial_means.detach().numpy()
        optimized_stds = initial_stds.detach().numpy()
        # Apply softmax to final logits to get the final weights
        optimized_weights = torch.softmax(initial_logits, dim=0).detach().numpy()

        return optimized_means, optimized_stds, optimized_weights



    @staticmethod
    def gaussian_mixture_pdf(x, meds, stds, weights):
        """
        Compute the PDF of a Gaussian Mixture Model.

        Parameters
        ----------
        x : array
            X values at which to compute the PDF.
        meds : list or array 
            Means for each Gaussian component.
        stds : list or array
            Standard deviations for each Gaussian component.
        weights : list or array
            Weights (relative importance that must sum to 1) for each Gaussian component.

        Returns
        -------
        pdf : array
            The computed PDF values for the Gaussian Mixture Model at each x.
        """
        # Ensure inputs are numpy arrays for consistent operations
        x = np.asarray(x)
        meds = np.asarray(meds)
        stds = np.asarray(stds)
        weights = np.asarray(weights)

        # Initialize the PDF to zero with the same shape as x
        pdf = np.zeros_like(x, dtype=float)

        # Compute the PDF for each Gaussian component and sum them up
        for med, std, weight in zip(meds, stds, weights):
            # Ensure std is positive to avoid issues with norm.pdf
            std = std if std > 0 else 1e-9
            # Compute the PDF of the individual component and add to the total, scaled by weight
            pdf += weight * norm.pdf(x, med, std)

        return pdf
    
    @staticmethod
    def sample_from_gmm(n_samples, means, stds, weights, random_state=None):
        """
        Samples a finite number of observations from a Gaussian Mixture Model (GMM).

        Parameters
        ----------
        n_samples : int
            The number of observations to sample.
        means : array 
            The means for each Gaussian component.
        stds : array
            The standard deviations for each Gaussian component.
        weights : array
            The weights (mixing proportions) for each Gaussian component. Weights should sum to 1.
        random_state : int, Generator, or None
            Controls the randomness for sampling. 
            If int, uses it as a seed. If Generator, uses it directly. 
            If None, uses the global random state (non-deterministic).

        Returns
        -------
        samples : array
            An array of sampled observations from the GMM.
        """
        # Ensure inputs are numpy arrays
        means = np.asarray(means)
        stds = np.asarray(stds)
        weights = np.asarray(weights)

        # Ensure weights sum to 1 (or normalize them)
        # Add a small epsilon to avoid division by zero if sum is very close to zero
        sum_weights = np.sum(weights)
        if sum_weights <= 1e-9:
            warnings.warn("GMM weights sum to zero or very close to zero. Cannot sample.", RuntimeWarning)
            return np.array([]) # Return empty array if weights are all zero
        elif not np.isclose(sum_weights, 1.0):
            warnings.warn("GMM weights do not sum to 1. Normalizing weights.", RuntimeWarning)
            weights = weights / sum_weights


        n_components = len(means)
        if not (len(stds) == n_components and len(weights) == n_components):
            raise ValueError("Number of means, standard deviations, and weights must be the same.")

        # Create a Generator instance based on random_state
        rng = np.random.default_rng(random_state)

        samples = np.zeros(n_samples)

        # For each sample, choose a component based on weights and draw from that component
        for i in range(n_samples):
            # Choose a component index based on weights
            # rng.choice supports weighted sampling
            component_index = rng.choice(n_components, p=weights)

            # Get parameters for the chosen component
            chosen_mean = means[component_index]
            chosen_std = stds[component_index]

            # Ensure standard deviation is positive for sampling
            chosen_std = chosen_std if chosen_std > 0 else 1e-9

            # Sample from the chosen normal distribution
            samples[i] = rng.normal(loc=chosen_mean, scale=chosen_std)

        return samples

