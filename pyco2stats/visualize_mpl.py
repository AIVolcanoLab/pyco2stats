import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from .sinclair import Sinclair
from .gaussian_mixtures import GMM 

class Visualize_Mpl:
    """
    Class for plotting Sinclair-style probability plots for raw data and GMMs.
    """

    @staticmethod
    def pp_raw_data(raw_data, ax=None, **scatter_kwargs):
        sigma_vals, sorted_data = Sinclair.raw_data_to_sigma(raw_data)
        if ax is None:
            fig, ax = plt.subplots()
        ax.scatter(sigma_vals, sorted_data, **scatter_kwargs)
        return ax

    @staticmethod
    def pp_combined_population(means, stds, weights, x_range=(-3.5, 3.5), ax=None, **line_kwargs):
        # Use extended x_vals to compute tails beyond the plot window
        x_vals = np.linspace(x_range[0] - 1.5, x_range[1] + 1.5, 600)
        y_cdf = Sinclair.combine_gaussians(x_vals, means, stds, weights)
        sigma_vals = Sinclair.cumulative_to_sigma(y_cdf)

        if ax is None:
            fig, ax = plt.subplots()

        # Just plot the full curve
        ax.plot(sigma_vals, x_vals, **line_kwargs)
        ax.set_xlim(x_range)
        return ax


    @staticmethod
    def pp_single_populations(means, stds, z_range=(-3.5, 3.5), ax=None, **line_kwargs):


        means = np.atleast_1d(means)
        stds  = np.atleast_1d(stds)

        for mean, std in zip(means, stds):
            Visualize_Mpl.pp_one_population(mean, std, z_range=(-3.5, 3.5), ax=ax, **line_kwargs)

        return ax  


    def pp_one_population(mean, std, z_range=(-3.5, 3.5), ax=None, **line_kwargs):
        z_vals = np.linspace(z_range[0], z_range[1], 600)

        if ax is None:
            fig, ax = plt.subplots()

        x_vals = mean + z_vals * std
        ax.plot(z_vals, x_vals, **line_kwargs)

        return ax   


    @staticmethod
    def pp_add_sigma_grid(ax=None, sigma_ticks=np.arange(-3, 4, 1)):
        if ax is None:
            fig, ax = plt.subplots()

        ax.xaxis.set_major_locator(ticker.FixedLocator(sigma_ticks))
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.set_xlim(-3.5, 3.5)
        return ax


    @staticmethod
    def pp_add_percentiles(ax=None, percentiles='standard', linestyle='-.', linewidth=1, color='green', label_size=10, **plot_kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        if percentiles == 'standard':
            perc_values = [1, 5, 10, 25, 50, 75, 95, 90, 99]
        elif percentiles == 'full':
            perc_values = [0.5, 1, 2, 4, 6, 8, 10, 15, 20, 25, 30, 35, 40, 50,
                           60, 65, 70, 75, 80, 85, 90, 92, 94, 96, 98, 99, 99.5]
        else:
            perc_values = percentiles

        sigma_ticks = norm.ppf(np.array(perc_values) / 100.0)
        ax_secondary = ax.secondary_xaxis('top')
        ax_secondary.set_xticks(sigma_ticks)
        #ax_secondary.set_xticklabels([f"{p:g}%" for p in perc_values], fontsize=label_size, rotation=90)
        ax_secondary.set_xticklabels([])

        for i, (perc, sigma) in enumerate(zip(perc_values, sigma_ticks)):
            ax.axvline(x=sigma, linestyle=linestyle, linewidth=linewidth, color=color, **plot_kwargs)
            y_offset = 1.01 + (i % 2) * 0.04 if percentiles == 'full' else 1.01
            ax.text(sigma, y_offset, f"{perc}", ha='center', va='bottom',
                    transform=ax.get_xaxis_transform(), fontsize=label_size, color='black')

        return ax


    @staticmethod
    def qq_plot(raw_data, model_data, ax, line_kwargs=None, marker_kwargs=None):
        
        """
        INSERIRE DESCRIZIONE

        Parameters:
        - ax (matplotlib.axes.Axes): The matplotlib Axes object where the quantiles will be plotted.
        - observed_data (array-like): The observationally derived data.
        - reference_population (array-like): The data referring to the reference population.

        Returns:
        - None: This function directly plots on the provided Axes object.
        """
        
        # Sort both observed data and reference population
        observed_data_sorted = np.sort(raw_data)
        reference_population_sorted = np.sort(model_data)

        # Number of data points
        n = len(observed_data_sorted)

        # Calculate the empirical percentiles for the observed data
        percentiles = np.linspace(0, 100, n)

        # Match the reference percentiles to the same empirical percentiles
        reference_percentiles = np.percentile(reference_population_sorted, percentiles)


        # Plot the observed data percentiles vs. reference population percentiles
        ax.plot(observed_data_sorted, reference_percentiles,  **marker_kwargs, linestyle='', label='Observed Data vs. Reference Population')

        # Plot the 45‑degree reference line
        # — remove the 'r--' fmt string, rely exclusively on line_kwargs
        # — default to color='r', linestyle='--' if user didn't pass any
        lk = line_kwargs or {}
        # ensure we don’t accidentally pass the fmt‑style redundant args
        ax.plot(
            [observed_data_sorted[0], observed_data_sorted[-1]],
            [observed_data_sorted[0], observed_data_sorted[-1]],
            **lk,
            label='45° Line'
        )

    def plot_gmm_pdf(ax, x, meds, stds, weights, data=None,
                 pdf_plot_kwargs=None, component_plot_kwargs=None, hist_plot_kwargs=None):
        """
        Plot the Gaussian Mixture Model PDF and its components.

        Parameters:
        - ax: Matplotlib axis object.
        - x (array): x values.
        - meds (list or array): Means of the Gaussian components.
        - stds (list or array): Standard deviations of the Gaussian components.
        - weights (list or array): Weights of the Gaussian components.
        - data (list or array , optional): Raw data to plot as a histogram.
        - pdf_plot_kwargs (list): Keyword arguments for the main GMM PDF plot.
        - component_plot_kwargs (list): Keyword arguments for the individual component plots.
        - hist_plot_kwargs (list): Keyword arguments for the histogram plot.
        """
        if pdf_plot_kwargs is None:
            pdf_plot_kwargs = {}
        if component_plot_kwargs is None:
            component_plot_kwargs = {}
        if hist_plot_kwargs is None:
            hist_plot_kwargs = {}

        # Compute the Gaussian Mixture PDF
        pdf = GMM.gaussian_mixture_pdf(x, meds, stds, weights)

        # Plot the Gaussian Mixture PDF
        ax.plot(x, pdf, label='Gaussian Mixture PDF', **pdf_plot_kwargs)

        # Plot each Gaussian component
        for i, (med, std, weight) in enumerate(zip(meds, stds, weights)):
            ax.plot(x, weight * norm.pdf(x, med, std), label=f'Component {i + 1}', **component_plot_kwargs)

        # Plot the histogram of the raw data if provided
        if data is not None:
            ax.hist(data, bins=20, density=True, **hist_plot_kwargs)

        ax.legend()
