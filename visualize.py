import numpy as np
from scipy.stats import norm
from .sinclair import Sinclair
from .gaussian_mixtures import GMM

class Visualize:
    
    @staticmethod
    def pp_raw_data(my_data, ax, **kwargs):
        """
        Plot the raw probability plot plot data on the given Axes object.

        Parameters:
        - my_data (array-like): The input data to be plotted in the QQ plot.
        - ax (matplotlib.axes.Axes): The matplotlib Axes object where the QQ plot will be drawn.
        - **kwargs: Additional keyword arguments passed to the scatter plot function.
        
        Returns:
        - None
        """
        # Get the theoretical quantiles and sorted data values
        osm, osr = Sinclair.get_raw_data(my_data)
        
        # Scatter plot of theoretical quantiles vs. ordered data values
        ax.scatter(osm, osr, **kwargs)

    @staticmethod
    def pp_combined_population(meds, stds, fds, mminy, mmaxy, ax, **kwargs):
        """
        Plot combined Gaussian mixture distribution on the provided Axes object.
        
        Parameters:
        - meds (list of array): Means for each Gaussian component.
        - stds (list of array): Standard deviations for each Gaussian component.
        - fds (list of array): Weights (relative importance) for each Gaussian component.
        - mminy (int or float): Minimum value for generating ylon values.
        - mmaxy (int or float): Maximum value for generating ylon values.
        - ax (matplotlib.axes.Axes): The matplotlib Axes object where the plot will be drawn.
        - **kwargs: Additional keyword arguments passed to the plot function.
        
        Returns:
        - None
        """
        # Calculate the sigma values and ylon values for the combined population
        x, y = Sinclair.calculate_combined_population(meds, stds, fds, mminy, mmaxy)
        
        # Plot the calculated sigma values against ylon values
        ax.plot(x, y, **kwargs)
        
    @staticmethod
    def pp_single_populations(meds, stds, mminy, mmaxy, ax, **kwargs):
        """
        Plot the individual Gaussian population curves on the provided Axes object.

        Parameters:
        - meds (list or array): means for each Gaussian component.
        - stds (list or array): standard deviations for each Gaussian component.
        - mminy (int or float): Minimum value for generating ylon values.
        - mmaxy (int or float): Maximum value for generating ylon values.
        - ax (matplotlib.axes.Axes): The Matplotlib Axes object where the curves will be plotted.
        - **kwargs: Additional keyword arguments passed to the plot function for styling.

        Returns:
        - None: This function directly plots on the provided Axes object.
        """
        # Initialize a counter for labeling each Gaussian component
        i = 0
        
        # Loop through each Gaussian component defined by its mean (med) and standard deviation (std)
        for med, std in zip(meds, stds):
            # Increment the counter to create a label for the current Gaussian component
            i += 1
            
            # Calculate the transformed sigma values and corresponding ylon values for the current Gaussian component
            # This is done by considering the current component as a single population with a weight of 1
            x, y = Sinclair.calculate_combined_population([med], [std], [1], mminy, mmaxy)
            
            # Plot the calculated values on the provided Axes object
            # The **kwargs allows for additional styling options such as color and linestyle
            ax.plot(x, y, **kwargs, label='Population ' + str(i))

        # Optionally, you might want to add a legend here to label each population curve
        # ax.legend()  # Uncomment this line if you want to automatically add a legend


    @staticmethod
    def pp_add_percentiles(ax, percentiles='full', label_size=8, **kwargs):
        """
        Plot vertical percentile lines on the provided Axes object and label them.

        Parameters:
        - ax (matplotlib.axes.Axes): The matplotlib Axes object where the percentiles will be plotted.
        - percentiles (str): The type of percentiles to plot ('full', 'half', 'even', 'deciles').
        - label_size (int): Font size for the labels.
        - **kwargs: Additional keyword arguments passed to the ax.axvline function.

        Returns:
        - None: This function directly plots on the provided Axes object.
        """
        peri = [-2.326, -2.055, -1.751, -1.555, -1.405, -1.282, -1.037, -0.842, -0.675, -0.525, -0.385, -0.255, 0,
                0.255, 0.385, 0.525, 0.675, 0.842, 1.037, 1.282, 1.405, 1.555, 1.751, 2.055, 2.326]
        perilab = [1, 2, 4, 6, 8, 10, 15, 20, 25, 30, 35, 40, 50, 60, 65, 70, 75, 80, 85, 90, 92, 94, 96, 98, 99]

        if percentiles == 'half':
            peri = [-2.326, -1.751, -1.405, -1.037, -0.675, -0.385, 0, 0.255, 0.525, 0.842, 1.282, 1.555, 2.055]
            perilab = [1, 4, 8, 15, 25, 35, 50, 65, 75, 85, 92, 96, 99]
        elif percentiles == 'even':
            peri = [-2.055, -1.751, -1.555, -1.405, -1.282, -0.842, -0.525, -0.255, 0, 0.255, 0.525, 0.842, 1.282, 1.405, 1.555, 1.751, 2.055]
            perilab = [2, 4, 6, 8, 10, 20, 30, 40, 50, 60, 70, 80, 90, 92, 94, 96, 98]
        elif percentiles == 'deciles':
            peri = [-1.282, -0.842, -0.525, -0.255, 0, 0.255, 0.525, 0.842, 1.282]
            perilab = [10, 20, 30, 40, 50, 60, 70, 80, 90]

        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(peri)

        if percentiles == 'full':
            labels = []
            for i, label in enumerate(perilab):
                if i % 2 == 0:
                    labels.append(f"\n{label}")  # Move even indices up
                else:
                    labels.append(f"{label}\n")  # Move odd indices down
            ax2.set_xticklabels(labels, fontsize=label_size)
        else:
            ax2.set_xticklabels(perilab, fontsize=label_size)

        for my_line in peri:
            ax.axvline(my_line, **kwargs)  # Draw vertical lines at the specified percentiles

    @staticmethod
    def qq_plot(ax, observed_data, reference_population):
        
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
        observed_data_sorted = np.sort(observed_data)
        reference_population_sorted = np.sort(reference_population)

        # Number of data points
        n = len(observed_data_sorted)

        # Calculate the empirical percentiles for the observed data
        percentiles = np.linspace(0, 100, n)

        # Match the reference percentiles to the same empirical percentiles
        reference_percentiles = np.percentile(reference_population_sorted, percentiles)


        # Plot the observed data percentiles vs. reference population percentiles
        ax.plot(observed_data_sorted, reference_percentiles, 'o', markersize=4, label='Observed Data vs. Reference Population')

        # Plot the 45-degree line for reference
        ax.plot([observed_data_sorted[0], observed_data_sorted[-1]], [observed_data_sorted[0], observed_data_sorted[-1]], 'r--', label='45-degree Line')

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

