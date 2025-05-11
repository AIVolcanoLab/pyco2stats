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
        """
        Plot a probability plot of raw data using Sinclair transformation.

        Parameters
        ----------
        raw_data : array-like
            Array of raw data values.
        ax : matplotlib.axes.Axes, optional
            Matplotlib Axes object to plot on. Creates new one if None.
        **scatter_kwargs : dict
            Additional keyword arguments passed to ax.scatter().

        Returns
        -------
        ax : matplotlib.axes.Axes
            The Axes object with the plot.
        """
        sigma_vals, sorted_data = Sinclair.raw_data_to_sigma(raw_data)
        if ax is None:
            fig, ax = plt.subplots()
        ax.scatter(sigma_vals, sorted_data, **scatter_kwargs)
        return ax

    @staticmethod
    def pp_combined_population(means, stds, weights, x_range=(-3.5, 3.5), ax=None, **line_kwargs):        
        """
        Plot the cumulative distribution of a Gaussian mixture model on a probability plot.

        Parameters
        ----------
        means : array-like
            Means of Gaussian components.
        stds : array-like
            Standard deviations of Gaussian components.
        weights : array-like
            Weights of each Gaussian component.
        x_range : tuple, optional
            Range of sigma-values (x-axis) to display.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. Creates new one if None.
        **line_kwargs : dict
            Additional arguments passed to ax.plot().

        Returns
        -------
        ax : matplotlib.axes.Axes
            The Axes object with the plot.
        """
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
        """
        Plot individual Gaussian distributions on a probability plot.

        Parameters
        ----------
        means : array-like
            Means of the Gaussian components.
        stds : array-like
            Standard deviations of the Gaussian components.
        z_range : tuple, optional
            Range of z-values to use for plotting.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. Creates new one if None.
        **line_kwargs : dict
            Additional arguments passed to ax.plot().

        Returns
        -------
        ax : matplotlib.axes.Axes
            The Axes object with the plots.
        """

        means = np.atleast_1d(means)
        stds  = np.atleast_1d(stds)

        for mean, std in zip(means, stds):
            Visualize_Mpl.pp_one_population(mean, std, z_range=(-3.5, 3.5), ax=ax, **line_kwargs)

        return ax  


    def pp_one_population(mean, std, z_range=(-3.5, 3.5), ax=None, **line_kwargs):
        """
        Plot a single Gaussian distribution on a probability plot.

        Parameters
        ----------
        mean : float
            Mean of the Gaussian distribution.
        std : float
            Standard deviation of the Gaussian distribution.
        z_range : tuple, optional
            Range of z-values to use for plotting.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. Creates new one if None.
        **line_kwargs : dict
            Additional arguments passed to ax.plot().

        Returns
        -------
        ax : matplotlib.axes.Axes
            The Axes object with the plot.
        """
        z_vals = np.linspace(z_range[0], z_range[1], 600)

        if ax is None:
            fig, ax = plt.subplots()

        x_vals = mean + z_vals * std
        ax.plot(z_vals, x_vals, **line_kwargs)

        return ax   


    @staticmethod
    def pp_add_sigma_grid(ax=None, sigma_ticks=np.arange(-3, 4, 1)):
        """
        Add vertical grid lines at specified sigma (z-score) positions.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to add the grid to. Creates new one if None.
        sigma_ticks : array-like
            Positions (z-scores) where grid lines should be added.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The Axes object with the updated grid.
        """
        if ax is None:
            fig, ax = plt.subplots()

        ax.xaxis.set_major_locator(ticker.FixedLocator(sigma_ticks))
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.set_xlim(-3.5, 3.5)
        return ax


    @staticmethod
    def pp_add_percentiles(ax=None, percentiles='standard', linestyle='-.', linewidth=1, color='green', label_size=10, **plot_kwargs):
        """
        Add percentile reference lines and labels to the top axis.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to annotate. Creates new one if None.
        percentiles : str or list, optional
            Which percentiles to use: 'standard', 'full', or custom list.
        linestyle : str
            Line style for vertical lines.
        linewidth : float
            Width of the percentile lines.
        color : str
            Color of percentile lines.
        label_size : int
            Font size of percentile labels.
        **plot_kwargs : dict
            Additional keyword arguments for ax.axvline().

        Returns
        -------
        ax : matplotlib.axes.Axes
            The Axes object with added percentile lines and labels.
        """
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
        Create a Q-Q plot comparing raw data to model-simulated data.

        Parameters
        ----------
        raw_data : array-like
            Observed dataset.
        model_data : array-like
            Simulated or reference dataset.
        ax : matplotlib.axes.Axes
            Axes object on which to draw the plot.
        line_kwargs : dict, optional
            Keyword arguments for the reference line.
        marker_kwargs : dict, optional
            Keyword arguments for the scatter points.

        Returns
        -------
        None
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

        Parameters
        ----------
        ax : Matplotlib axis object
            Axes object where to plot.
        x : array
            x values.
        meds : list or array
            Means of the Gaussian components.
        stds : list or array
            Standard deviations of the Gaussian components.
        weights : list or array
            Weights of the Gaussian components.
        data : list or array, optional 
            Raw data to plot as a histogram.
        pdf_plot_kwargs : list
            Keyword arguments for the main GMM PDF plot.
        component_plot_kwargs : list 
            Keyword arguments for the individual component plots.
        hist_plot_kwargs : list
             Keyword arguments for the histogram plot.

        Returns
        -------
        None
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
