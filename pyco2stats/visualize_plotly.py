
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from .sinclair import Sinclair
from .gaussian_mixtures import GMM

class Visualize_Plotly:
    """
    Plotly-based Sinclair-style probability plots for raw data and GMMs.
    """

    @staticmethod
    def pp_raw_data(raw_data, fig=None, marker_kwargs=None):
        """
        Plot raw data on log-normal probability paper:
          x = sigma-values (z), y = sorted data.

        You may pass marker_kwargs as either:
          - top-level: size, color, opacity, symbol
          - or a nested dict under "marker"
        """
        # 1) compute the sigma‐quantiles and sort
        sigma_vals, sorted_data = Sinclair.raw_data_to_sigma(raw_data)

        # 2) copy the user kwargs so we don't mutate their dict
        margs = {} if marker_kwargs is None else marker_kwargs.copy()

        # 3) extract any nested marker dict
        marker = margs.pop('marker', {}).copy()

        # 4) also move any top‑level size/color/etc into that dict
        for key in ('size', 'color', 'opacity', 'symbol'):
            if key in margs:
                marker[key] = margs.pop(key)

        # 5) if we collected any marker attributes, re‑nest them
        if marker:
            margs['marker'] = marker

        # 6) build the scatter trace
        trace = go.Scatter(
            x=sigma_vals,
            y=sorted_data,
            mode='markers',
            name='Raw Data',
            **margs
        )

        # 7) add to figure if given
        if fig is not None:
            fig.add_trace(trace)

        return trace

    @staticmethod
    def qq_plot(raw_data, fig=None, scatter_kwargs=None, line_kwargs=None):
        """
        Generate a QQ-plot: theoretical sigma quantiles vs sorted data.
        Returns: (scatter trace, line trace).
        """
        sigma_vals, sorted_data = Sinclair.raw_data_to_sigma(raw_data)
        skw = scatter_kwargs or {}
        scatter = go.Scatter(
            x=sigma_vals,
            y=sorted_data,
            mode='markers',
            name='QQ Plot',
            **skw
        )
        # fit reference line
        slope, intercept = np.polyfit(sigma_vals, sorted_data, 1)
        x_line = np.array([sigma_vals.min(), sigma_vals.max()])
        y_line = intercept + slope * x_line
        lkw = line_kwargs or {}
        line = go.Scatter(
            x=x_line,
            y=y_line,
            mode='lines',
            name='Fit Line',
            line=lkw
        )
        if fig is not None:
            fig.add_trace(scatter)
            fig.add_trace(line)
        return scatter, line

    @staticmethod
    def pp_one_population(mean, std, fig=None, z_range=(-3.5,3.5), line_kwargs=None):
        """
        Plot a single Gaussian population on probability paper.
        Returns: the line trace.
        """
        z_vals = np.linspace(z_range[0], z_range[1], 600)
        x_vals = mean + z_vals * std
        largs = line_kwargs or {}
        trace = go.Scatter(
            x=z_vals,
            y=x_vals,
            mode='lines',
            name=f'Pop μ={mean:.2f}, σ={std:.2f}',
            line=largs
        )
        if fig is not None:
            fig.add_trace(trace)
        return trace

    @staticmethod
    def pp_single_populations(means, stds, fig=None, z_range=(-3.5,3.5), line_kwargs=None):
        """
        Plot each Gaussian component as separate line.
        Returns: list of line traces.
        """
        means = np.atleast_1d(means)
        stds  = np.atleast_1d(stds)
        traces = []
        for mean, std in zip(means, stds):
            tr = Visualize_Plotly.pp_one_population(mean, std, fig, z_range, line_kwargs)
            traces.append(tr)
        return traces

    @staticmethod
    def pp_combined_population(means, stds, weights, fig=None, z_range=(-3.5,3.5), line_kwargs=None):
        """
        Plot the combined mixture CDF, transformed to sigma-values.
        Returns: the line trace.
        """
        x = np.linspace(z_range[0], z_range[1], 600)
        cdf = Sinclair.combine_gaussians(x, np.array(means), np.array(stds), np.array(weights))
        sigma_vals = Sinclair.cumulative_to_sigma(cdf)
        largs = line_kwargs or {}
        trace = go.Scatter(
            x=sigma_vals,
            y=x,
            mode='lines',
            name='Combined Population',
            line=largs
        )
        if fig is not None:
            fig.add_trace(trace)
        return trace

    @staticmethod
    def pp_add_percentiles(
        fig,
        percentiles="full",
        line_kwargs: dict = None,
        label_kwargs: dict = None,
        y_min: float = None,
        y_max: float = None
    ):
        """
        Add vertical percentile lines and labels at the top.

        Parameters
        ----------
        fig : plotly.graph_objects.Figure
        percentiles : 'full' or iterable of ints
        line_kwargs : dict for fig.add_shape(line=...)
        label_kwargs: dict for fig.add_annotation(...)
        y_min, y_max : if provided, override y-axis span
        """
        # 1) choose levels
        levels = [1,5,10,25,50,75,90,95,99] if percentiles=="full" else list(percentiles)

        # 2) infer y-span
        if y_min is None or y_max is None:
            rng = fig.layout.yaxis.range or []
            if len(rng)==2:
                y_min, y_max = rng
            else:
                y_min, y_max = 0, 1

        # 3) defaults
        lkw = line_kwargs or dict(color="#c8c8c8", dash="dash", width=0.5)
        awk = label_kwargs or dict(font=dict(size=12,color="#555555"),
                                   showarrow=False, yshift=8)

        # 4) draw each
        from scipy.stats import norm
        for p in levels:
            z = norm.ppf(p/100.0)
            fig.add_shape(
                type="line",
                x0=z, x1=z,
                y0=y_min, y1=y_max,
                line=lkw,
                layer="below"
            )
            fig.add_annotation(
                x=z, y=y_max,
                text=f"{p}%",
                xanchor="center",
                **awk
            )
    @staticmethod
    def plot_gmm_pdf(x_values, meds, stds, weights,
                     data=None,
                     pdf_plot_kwargs=None,
                     component_plot_kwargs=None,
                     hist_plot_kwargs=None):
        """
        Generate Plotly traces for a Gaussian mixture PDF:
          1) Histogram of raw data (probability density)
          2) Individual component PDFs
          3) Total mixture PDF

        Parameters:
        - x_values: 1D numpy array of x points for evaluating the PDFs.
        - meds, stds, weights: lists/arrays of means, stds, and mixture weights.
        - data: raw data array for histogram (optional).
        - pdf_plot_kwargs: dict of kwargs for the mixture PDF line.
        - component_plot_kwargs: dict of kwargs for each component line.
        - hist_plot_kwargs: dict of kwargs for the histogram; may include:
            • 'bins' (int or 'auto') to control bin count/method,
            • any other go.Histogram properties.

        Returns:
        - hist_trace: Histogram trace or None.
        - comp_traces: List of component line traces.
        - pdf_trace: Total mixture PDF line trace.
        """
        import numpy as _np
        from scipy.stats import norm
        import plotly.graph_objects as go
        from .gaussian_mixtures import GMM

        # prepare kwargs dicts
        pdf_plot_kwargs       = pdf_plot_kwargs or {}
        component_plot_kwargs = component_plot_kwargs or {}
        hist_plot_kwargs      = hist_plot_kwargs or {}

        # 1) histogram trace
        hist_trace = None
        if data is not None:
            # copy so we don't pop from the user's dict
            hkwargs = hist_plot_kwargs.copy()
            bins = hkwargs.pop('bins', 20)

            if bins == 'auto':
                # compute edges via numpy 'auto'
                edges = _np.histogram_bin_edges(data, bins='auto')
                size = edges[1] - edges[0]
                hargs = {
                    'xbins': dict(start=edges[0], end=edges[-1], size=size),
                    'histnorm': 'probability density'
                }
            else:
                # fixed number of bins
                hargs = {
                    'nbinsx': bins,
                    'histnorm': 'probability density'
                }

            # merge any remaining user kwargs
            hargs.update(hkwargs)
            hist_trace = go.Histogram(x=data, **hargs)

        # 2) individual component traces
        comp_traces = []
        for idx, (mu, sigma, w) in enumerate(zip(meds, stds, weights), start=1):
            y_comp = w * norm.pdf(x_values, mu, sigma)
            comp_traces.append(
                go.Scatter(
                    x=x_values,
                    y=y_comp,
                    mode='lines',
                    name=f'Component {idx}',
                    **component_plot_kwargs
                )
            )

        # 3) total mixture PDF
        try:
            # if your GMM class defines this
            pdf_vals = GMM.gaussian_mixture_pdf(x_values, meds, stds, weights)
        except Exception:
            # fallback manual sum
            pdf_vals = _np.zeros_like(x_values, dtype=float)
            for mu_i, sigma_i, w_i in zip(meds, stds, weights):
                pdf_vals += w_i * norm.pdf(x_values, mu_i, sigma_i)

        pdf_trace = go.Scatter(
            x=x_values,
            y=pdf_vals,
            mode='lines',
            name='Gaussian Mixture PDF',
            **pdf_plot_kwargs
        )

        return hist_trace, comp_traces, pdf_trace

    @staticmethod
    def qq_plot(raw_data, model_data, fig=None,
                    marker_kwargs=None, line_kwargs=None):
            """
            Draw a Q–Q plot comparing two samples:
            - raw_data: 1D array of empirical observations
            - model_data: 1D array of simulated/fitted data
            marker_kwargs: dict passed to go.Scatter for the sample points
            line_kwargs:   dict passed to go.Scatter for the y=x reference line
            Returns the two traces.
            """
            import numpy as np
            from scipy.stats import norm
            import plotly.graph_objects as go

            # 1) compute matching percentiles
            probs   = np.linspace(0, 1, len(raw_data))
            q_raw   = np.quantile(raw_data,   probs)
            q_model = np.quantile(model_data, probs)

            # 2) the Q–Q scatter
            mk = marker_kwargs or {}
            pts = go.Scatter(
                x=q_raw, y=q_model,
                mode="markers",
                name="Q–Q points",
                **mk
            )
            if fig is not None:
                fig.add_trace(pts)

            # 3) identity line
            mn = min(q_raw.min(), q_model.min())
            mx = max(q_raw.max(), q_model.max())
            lk = line_kwargs or {}
            line = go.Scatter(
                x=[mn, mx], y=[mn, mx],
                mode="lines",
                name="y = x",
                **lk
            )
            if fig is not None:
                fig.add_trace(line)

            return pts, line

