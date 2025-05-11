import numpy as np
import warnings
import torch # Constrained GMM uses PyTorch
from sklearn.exceptions import ConvergenceWarning # sklearn GMM might raise this
from pyco2stats.gaussian_mixtures import GMM
from tqdm import tqdm

class Propagate_Errors:
    """
    A class to perform Monte Carlo error propagation for Gaussian Mixture Models (GMMs)
    fitted using different methods. Assumes input data is log-transformed.
    Propagates errors by adding fixed-standard-deviation additive noise to log-transformed data,
    corresponding to relative error on the original scale. Includes parameter alignment.
    """

    @staticmethod
    def _generate_perturbed_sample(data_log_scale, percentage_relative_error):
        """
        Generates a single perturbed dataset by adding random noise to the
        log-transformed data, where the noise standard deviation is a fixed
        value determined by the percentage relative error on the original scale.

        Parameters
        ----------
        data_log_scale : array or list
            The original 1D log-transformed input data. Will be converted to a NumPy array.
        percentage_relative_error : float
            The relative uncertainty in percent (e.g., 5 for 5%).
            This directly determines the standard deviation of the additive noise on the log scale.

        Returns
        -------
        Result : array
            A new array with perturbed data points on the log scale.
        """
        # Ensure data is a numpy array
        data_log_scale = np.asarray(data_log_scale)

        # Calculate the standard deviation of the additive noise on the log scale
        # This is the percentage relative error (as a fraction)
        std_dev_additive_noise = percentage_relative_error / 100.0

        # Generate random noise from a normal distribution with the calculated std dev
        additive_noise = np.random.normal(loc=0, scale=std_dev_additive_noise, size=data_log_scale.shape)

        # Add the noise to the log-transformed data
        perturbed_data_log_scale = data_log_scale + additive_noise

        return perturbed_data_log_scale

    @staticmethod
    def propagate_em_error(
        original_log_data, # Renamed to clarify it's the log data, removed type hint for flexibility
        percentage_relative_error: float, # Renamed parameter for clarity
        n_simulations: int,
        n_components: int,
        random_state: None,
        max_iter: int = 100,
        tol: float = 1e-6,
        show_progress: bool = False
    ) -> dict:
        """
        Propagates error through the EM-based GMM fitting by simulating
        additive noise on the log-transformed sample data using Monte Carlo.
        Noise std dev on log scale is fixed, equal to percentage_relative_error / 100.
        Aligns components by sorting means before storing results.
        Returns perturbed data statistics for diagnostics.

        Parameters
        ----------
        original_log_data : array
            The original 1D log-transformed data.
        percentage_relative_error : float
            The relative uncertainty in percent (e.g., 5 for 5%). This sets the std dev of additive noise on log scale.
        n_simulations : int
            The number of Monte Carlo simulations to run.
        n_components : int
            The number of Gaussian components in the mixture.
        max_iter : int
            Max iterations for the EM algorithm.
        tol : float
            Tolerance for EM convergence.

        Returns
        -------
        Result : dist
            A dictionary containing lists of results from each simulation:
            {'means': list[np.ndarray], 'std_devs': list[np.ndarray], 'weights': list[np.ndarray],
            'perturbed_data_means': list[float], 'perturbed_data_stds': list[float]}
            Lists of GMM parameters have shape (n_components,).
            Lists of data statistics have shape (n_simulations,).
        """
        # Ensure input data is a numpy array
        original_log_data = np.asarray(original_log_data)

        simulated_means = []
        simulated_std_devs = []
        simulated_weights = []
        perturbed_data_means = [] # To store mean of each perturbed sample
        perturbed_data_stds = []  # To store std dev of each perturbed sample


        # original_log_data_std and original_log_data_mean are not needed for perturbation std dev calculation now


       
        iterator = tqdm(range(n_simulations), desc="gaussian_mixture_em Monte Carlo") if show_progress else range(n_simulations)
        for i in iterator:
            # Generate perturbed log-transformed data using additive noise
            # Pass original_log_data and percentage_relative_error directly
            perturbed_log_data = Propagate_Errors._generate_perturbed_sample(
                original_log_data, percentage_relative_error # Pass percentage directly
            )

            # Store stats of the perturbed sample BEFORE fitting GMM
            perturbed_data_means.append(np.mean(perturbed_log_data))
            perturbed_data_stds.append(np.std(perturbed_log_data))

            # Catch potential warnings or errors from GMM fitting on perturbed data
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=ConvergenceWarning)
                try:
                    # Call the EM-based GMM fitting function
                    means, std_devs, weights, _ = GMM.gaussian_mixture_em(
                        perturbed_log_data, n_components, max_iter=max_iter, random_state=random_state, tol=tol
                    )

                    # --- Parameter Alignment Step ---
                    means = np.asarray(means)
                    std_devs = np.asarray(std_devs)
                    weights = np.asarray(weights)

                    sort_indices = np.argsort(means)
                    aligned_means = means[sort_indices]
                    aligned_std_devs = std_devs[sort_indices]
                    aligned_weights = weights[sort_indices]
                    # --- End Alignment Step ---

                    simulated_means.append(aligned_means)
                    simulated_std_devs.append(aligned_std_devs)
                    simulated_weights.append(aligned_weights)
                except Exception as e:
                    warnings.warn(f"Simulation {i+1}/{n_simulations} failed during GMM.gaussian_mixture_em: {e}", RuntimeWarning)
                    perturbed_data_means.pop() # Remove the last added mean for failed sim
                    perturbed_data_stds.pop()  # Remove the last added std dev for failed sim


        return {
            'means': simulated_means,
            'std_devs': simulated_std_devs,
            'weights': simulated_weights,
            'perturbed_data_means': perturbed_data_means, # Return the list of perturbed data means for successful fits
            'perturbed_data_stds': perturbed_data_stds   # Return the list of perturbed data stds for successful fits
        }

    @staticmethod
    def propagate_sklearn_error(
        original_log_data, # Renamed, removed type hint
        percentage_relative_error: float, # Renamed parameter
        n_simulations: int,
        n_components: int,
        max_iter: int = 10,
        tol: float = 1e-10,
        n_init: int = 20,
        suppress_warnings: bool = True,
        covariance_type: str = 'spherical',
        show_progress: bool = False
    ) -> dict:
        """
        Propagates error through the sklearn-based GMM fitting by simulating
        additive noise on the log-transformed sample data using Monte Carlo.
        Noise std dev on log scale is fixed, equal to percentage_relative_error / 100.
        Aligns components by sorting means before storing results.

        Parameters
        ----------
        original_log_data : array
            The original 1D log-transformed data.
        percentage_relative_error : float
            The relative uncertainty in percent (e.g., 5 for 5%). This sets the std dev of additive noise on log scale.
        n_simulations : int
            The number of Monte Carlo simulations to run.
        n_components : int
            The number of Gaussian components in the mixture.
        max_iter : int
            Max iterations for the sklearn EM algorithm.
        tol : float
            Tolerance for sklearn EM convergence.
        n_init : int
            Number of initializations for sklearn GMM.
        suppress_warnings : bool
            Whether to suppress sklearn warnings.
        covariance_type : string
            The type of covariance ('spherical', 'diag', 'full', 'tied').

        Returns
        -------
        result : dict
            A dictionary containing lists of results from each simulation:
            {'means': list[np.ndarray], 'std_devs': list[np.ndarray], 'weights': list[np.ndarray]}
            Each np.ndarray in the lists has shape (n_components,).
        """
        # Ensure input data is a numpy array
        original_log_data = np.asarray(original_log_data)


        simulated_means = []
        simulated_std_devs = []
        simulated_weights = []

        # original_log_data_std is not needed for perturbation std dev calculation now


        iterator = tqdm(range(n_simulations), desc="gaussian_mixture_sklearn Monte Carlo") if show_progress else range(n_simulations)
        for i in iterator:
            # Generate perturbed log-transformed data using additive noise
            # Pass original_log_data and percentage_relative_error directly
            perturbed_log_data = Propagate_Errors._generate_perturbed_sample(
                original_log_data, percentage_relative_error # Pass percentage directly
            )

            # sklearn GMM expects data in a 2D array, even for univariate data
            perturbed_data_2d = perturbed_log_data.reshape(-1, 1)

            # Call the sklearn-based GMM fitting function
            try:
                means, std_devs, weights, _ = GMM.gaussian_mixture_sklearn(
                    perturbed_data_2d, n_components=n_components, max_iter=max_iter,
                    tol=tol, n_init=n_init, suppress_warnings=suppress_warnings,
                    covariance_type=covariance_type
                )

                # --- Parameter Alignment Step ---
                means = np.asarray(means)
                std_devs = np.asarray(std_devs)
                weights = np.asarray(weights)

                sort_indices = np.argsort(means)
                aligned_means = means[sort_indices]
                aligned_std_devs = std_devs[sort_indices]
                aligned_weights = weights[sort_indices]
                # --- End Alignment Step ---

                simulated_means.append(aligned_means)
                simulated_std_devs.append(aligned_std_devs)
                simulated_weights.append(aligned_weights)

            except Exception as e:
                 warnings.warn(f"Simulation {i+1}/{n_simulations} failed during GMM.gaussian_mixture_sklearn: {e}", RuntimeWarning)
                 # Skip failed simulations


        return {
            'means': simulated_means,
            'std_devs': simulated_std_devs, 
            'weights': simulated_weights
        }


    @staticmethod
    def propagate_constrained_error(
        original_log_data, # Renamed, removed type hint
        percentage_relative_error: float, # Renamed parameter
        n_simulations: int,
        mean_constraints: list,
        std_constraints: list,
        n_components: int,
        n_epochs: int = 5000,
        lr: float = 0.001,
        verbose: bool = False, # Suppress verbose output during MC simulations
        show_progress: bool = False
    ) -> dict:
        """
        Propagates error through the constrained PyTorch-based GMM fitting by simulating
        additive noise on the log-transformed sample data using Monte Carlo.
        Noise std dev on log scale is fixed, equal to percentage_relative_error / 100.
        Aligns components by sorting means before storing results.

        Parameters
        ----------
        original_log_data : array
            The original 1D log-transformed data.
        percentage_relative_error : float
            The relative uncertainty in percent (e.g., 5 for 5%). This sets the std dev of additive noise on log scale.
        n_simulations : int
            The number of Monte Carlo simulations to run.
        mean_constraints : list
            List of tuples specifying (min, max) constraints for each component's mean on the log scale.
        std_constraints : list
            List of tuples specifying (min, max) constraints for each component's std dev on the log scale.
        n_components : int
            Number of Gaussian components.
        n_epochs : int
            Number of optimization epochs for constrained GMM.
        lr : float
            Learning rate for optimization.
        verbose : bool
            Whether to print progress during constrained GMM fitting (False recommended for MC).

        Returns
        -------
        result: dict
            A dictionary containing lists of results from each simulation:
            {'means': list[np.ndarray], 'std_devs': list[np.ndarray], 'weights': list[np.ndarray]}
            Each np.ndarray in the lists has shape (n_components,).
        """
        # Ensure input data is a numpy array
        original_log_data = np.asarray(original_log_data)


        simulated_means = []
        simulated_std_devs = []
        simulated_weights = []

        # original_log_data_std is not needed for perturbation std dev calculation now


        iterator = tqdm(range(n_simulations), desc="constrained_gaussian_mixture Monte Carlo") if show_progress else range(n_simulations)
        for i in iterator:
            # Generate perturbed log-transformed data using additive noise
            # Pass original_log_data and percentage_relative_error directly
            perturbed_log_data = Propagate_Errors._generate_perturbed_sample(
                original_log_data, percentage_relative_error # Pass percentage directly
            )

            # Convert perturbed data (NumPy array) to PyTorch tensor using torch.from_numpy
            perturbed_data_tensor = torch.from_numpy(perturbed_log_data).float()


            # Call the constrained PyTorch GMM fitting function
            try:
                means, std_devs, weights = GMM.constrained_gaussian_mixture(
                    perturbed_data_tensor, mean_constraints, std_constraints,
                    n_components, n_epochs=n_epochs, lr=lr, verbose=False # Force verbose off for MC
                )

                # --- Parameter Alignment Step ---
                # Note: PyTorch tensors need detachment before sorting with numpy
                means_np = means.detach().numpy()
                sort_indices = np.argsort(means_np)

                aligned_means = means_np[sort_indices]
                # Need to align std_devs and weights (which are already numpy from GMM method)
                aligned_std_devs = std_devs[sort_indices]
                aligned_weights = weights[sort_indices]
                # --- End Alignment Step ---

                simulated_means.append(aligned_means)
                simulated_std_devs.append(aligned_std_devs)
                simulated_weights.append(aligned_weights)

            except Exception as e:
                warnings.warn(f"Simulation {i+1}/{n_simulations} failed during GMM.constrained_gaussian_mixture: {e}", RuntimeWarning)
                # Skip failed simulations


        return {
            'means': simulated_means,
            'std_devs': simulated_std_devs,
            'weights': simulated_weights
        }

    @staticmethod
    def elaborate_results(
        propagation_results: dict,
        single_fit_means: np.ndarray = None,
        single_fit_std_devs: np.ndarray = None,
        single_fit_weights: np.ndarray = None,
        original_data_mean: float = None,
        original_data_std: float = None,
        method_name: str = "GMM" # e.g., "EM", "Sklearn", "Constrained"
    ):
        """
        Elaborates and prints the results from a Monte Carlo error propagation run.

        Parameters
        ----------
        propagation_results : dict
            The dictionary returned by a propagate_*_error method. Expected keys: 'means', 'std_devs', 'weights'.
            May also contain 'perturbed_data_means', 'perturbed_data_stds' for certain methods (e.g., EM).
        single_fit_means : array, optional
            Means from a single fit on original data.
        single_fit_std_devs : array, optional
            Std devs from a single fit on original data.
        single_fit_weights : array, optional
            Weights from a single fit on original data.
        original_data_mean : float, optional
            Mean of the original log-transformed data.
        original_data_std : float, optional
            Std dev of the original log-transformed data.
        method_name : string
            The name of the GMM method used for reporting.

        Returns
        -------
        None
        """
        print(f"\n--- {method_name} Propagation Results (Elaborated) ---")

        if not propagation_results['means']:
            print(f"No successful {method_name} GMM simulations.")
            return

        simulated_means = np.array(propagation_results['means'])
        simulated_std_devs = np.array(propagation_results['std_devs'])
        simulated_weights = np.array(propagation_results['weights'])

        n_simulations = len(simulated_means) # Get actual number of successful simulations

        # Calculate central tendency (mean and median) and confidence intervals (2.5% - 97.5%)
        mean_means = np.mean(simulated_means, axis=0)
        median_means = np.median(simulated_means, axis=0)
        ci_means = np.percentile(simulated_means, [2.5, 97.5], axis=0)

        mean_std_devs = np.mean(simulated_std_devs, axis=0)
        median_std_devs = np.median(simulated_std_devs, axis=0)
        ci_std_devs = np.percentile(simulated_std_devs, [2.5, 97.5], axis=0)

        mean_weights = np.mean(simulated_weights, axis=0)
        median_weights = np.median(simulated_weights, axis=0)
        ci_weights = np.percentile(simulated_weights, [2.5, 97.5], axis=0)


        print(f"  Successful Simulations: {n_simulations}")
        print(f"  Mean Means (log scale): {mean_means}")
        print(f"  Median Means (log scale): {median_means}")
        print(f"  CI (2.5% - 97.5%) Means (log scale): {ci_means}")
        print(f"  Mean Std Devs (log scale): {mean_std_devs}")
        print(f"  Median Std Devs (log scale): {median_std_devs}")
        print(f"  CI (2.5% - 97.5%) Std Devs (log scale): {ci_std_devs}")
        print(f"  Mean Weights: {mean_weights}")
        print(f"  Median Weights: {median_weights}")
        print(f"  CI (2.5% - 97.5%) Weights: {ci_weights}")


        
