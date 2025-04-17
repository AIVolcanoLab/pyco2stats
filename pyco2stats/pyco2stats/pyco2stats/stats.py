import numpy as np
import scipy.special as sp
import scipy.stats as stats
import statsmodels.api as sm

from scipy.stats.mstats import trim as scipy_trim
from scipy.stats import trimboth as scipy_trimboth
from scipy.stats.mstats import trimtail as scipy_trimtail
from scipy.stats import tmean as scipy_tmean
from scipy.stats.mstats import trimmed_std as scipy_trimmed_std 
from scipy.stats.mstats import winsorize as scipy_winsorize

from astropy.stats import biweight_location as astropy_biweight_location
from astropy.stats import biweight_scale as astropy_biweight_scale
from astropy.stats import median_absolute_deviation as astropy_median_absolute_deviation
from astropy.stats import mad_std as astropy_mad_std
from astropy.stats import sigma_clip as astropy_sigma_clip
from astropy.stats import sigma_clipped_stats as astropy_sigma_clipped_stats

class Stats:
    @staticmethod
    def sample_from_pdf(x, pdf, n_samples=1000):
        """
        Sample data points from a given probability density function (PDF).

        Parameters:
        - x (array): x values corresponding to the PDF.
        - pdf (array): PDF values corresponding to each x value.
        - n_samples (int): Number of samples to generate. Default is 1000.

        Returns:
        - array: sampled data points.
        """
        # Ensure the PDF is normalized (sum equals 1)
        pdf = pdf / np.sum(pdf)

        # Step 1: Create the Cumulative Distribution Function (CDF) from the PDF
        cdf = np.cumsum(pdf)
        
        # Step 2: Generate random values uniformly distributed between 0 and 1
        random_values = np.random.rand(n_samples)

        # Step 3: Use the CDF to map these random values to the corresponding x values
        # Find the indices where the random values fit in the CDF
        sample_indices = np.searchsorted(cdf, random_values)

        # Get the x values corresponding to these indices
        sampled_data = x[sample_indices]
        
        return sampled_data

    @staticmethod
    def sichel_function(z , n , M):
        
        """
        Compute Sichel's function value. From the formula:
        
        .. math::
        f(z|n,M) = 1 + \frac{z(n-1)}{n} + \sum^{M}_{m=2} \frac{z^{m}(n-1)^{2m-1}}{n^{m}m!\prod_{k=1}^{m-1}(n+2k-1)}
        
        Parameters:
        z (float): The argument to the Sichel function.
        n (int): The sample size.
        M (int): The order of the sSichel function.

        Returns:
        float: The value of Sichel's function.
        """

        sichel = 1. + z*(n-1)/n

        for m in range(2,M+1):

            A = ((z/n)**m)*((n-1)**(2*m-1))/sp.factorial(m)

            B = 1

            for k in range(1,m):

                B *= n + 2*k - 1

            sichel += A/B
            
        return sichel
        

    #@staticmethod
    #def sichel_function_15(z, n, M):
    #    """
    #    Compute Sichel's function using 15 terms.
    #    
    #    Parameters:
    #    z (float): The argument to the Sichel function.
    #    n (int): The sample size.
    #
    #    Returns:
    #    float: The value of Sichel's function.
    #    """
    #    sf_15 = (1 + z * (n-1)/n +
    #             ((z**2) * (n-1)**3)/(n**2*(n+1)*sp.factorial(2)) +
    #             ((z**3) * (n-1)**5)/(n**3*(n+1)*(n+3)*sp.factorial(3)) +
    #             ((z**4) * (n-1)**7)/(n**4*(n+1)*(n+3)*(n+5)*sp.factorial(4)) +
    #             ((z**5) * (n-1)**9)/(n**5*(n+1)*(n+3)*(n+5)*(n+7)*sp.factorial(5)) +
    #             ((z**6) * (n-1)**11)/(n**6*(n+1)*(n+3)*(n+5)*(n+7)*(n+9)*sp.factorial(6)) +
    #             ((z**7) * (n-1)**13)/(n**7*(n+1)*(n+3)*(n+5)*(n+7)*(n+9)*(n+11)*sp.factorial(7)) +
    #             ((z**8) * (n-1)**15)/(n**8*(n+1)*(n+3)*(n+5)*(n+7)*(n+9)*(n+11)*(n+13)*sp.factorial(8)) +
    #             ((z**9) * (n-1)**17)/(n**9*(n+1)*(n+3)*(n+5)*(n+7)*(n+9)*(n+11)*(n+13)*(n+15)*sp.factorial(9)) +
    #             ((z**10) * (n-1)**19)/(n**10*(n+1)*(n+3)*(n+5)*(n+7)*(n+9)*(n+11)*(n+13)*(n+15)*(n+17)*sp.factorial(10)) +
    #             ((z**11) * (n-1)**21)/(n**11*(n+1)*(n+3)*(n+5)*(n+7)*(n+9)*(n+11)*(n+13)*(n+15)*(n+17)*(n+19)*sp.factorial(11)) +
    #             ((z**12) * (n-1)**23)/(n**12*(n+1)*(n+3)*(n+5)*(n+7)*(n+9)*(n+11)*(n+13)*(n+15)*(n+17)*(n+19)*(n+21)*sp.factorial(12)) +
    #             ((z**13) * (n-1)**25)/(n**13*(n+1)*(n+3)*(n+5)*(n+7)*(n+9)*(n+11)*(n+13)*(n+15)*(n+17)*(n+19)*(n+21)*(n+23)*sp.factorial(13)) +
    #             ((z**14) * (n-1)**27)/(n**14*(n+1)*(n+3)*(n+5)*(n+7)*(n+9)*(n+11)*(n+13)*(n+15)*(n+17)*(n+19)*(n+21)*(n+23)*(n+25)*sp.factorial(14)) +
    #             ((z**15) * (n-1)**29)/(n**15*(n+1)*(n+3)*(n+5)*(n+7)*(n+9)*(n+11)*(n+13)*(n+15)*(n+17)*(n+19)*(n+21)*(n+23)*(n+25)*(n+27)*sp.factorial(15)))
    #    return sf_15

    @staticmethod
    def sichel_function_log(sigma_sq, n, max_terms=20, tol=1e-10):
        """
        Compute Sichel's function ψ_n using logarithmic space for stability.

        Parameters:
        sigma_sq (float): The variance of the log-transformed data.
        n (int): The sample size.
        max_terms (int): Maximum number of terms to include in the series. Default is 20.
        tol (float): Tolerance for convergence. Default is 1e-10.

        Returns:
        psi_n (float): The value of Sichel's function.
        """
        psi_n = 1 + (n - 1) / n * sigma_sq  # Start with the first two terms
        for j in range(2, max_terms + 1):
            log_numerator = (2 * j - 1) * np.log(n - 1)
            log_denominator = j * np.log(n) + np.sum(np.log(np.arange(n + 1, n + 2 * j, 2)))
            log_term = log_numerator - log_denominator + j * np.log(sigma_sq) - sp.gammaln(j + 1)
            term = np.exp(log_term)
            psi_n += term
            if term < tol:
                break
        return psi_n

    @staticmethod
    def mvue_lnorm_dist(data, confidence=0.95, transformed=False, sighel="log", M = 15):
        """
        Calculate the Minimum Variances Unbiased Estimator mean and confidence interval for log-normally distributed data using Sichel's function.
        
        Parameters:
        data (array-like): Log-normally distributed data.
        confidence (float): Confidence level for the interval. Default is 0.95.
        transformed (bool): if True, it assumes that data are ln transformed. Default is False.

        Returns:
        dict: A dictionary containing the MVUE mean and confidence interval.
        """
        # Step 1: Transform the data using the natural log if not already transformed
        if not transformed:
            log_data = np.log(data)
        else:
            log_data = data

        # Step 2: Calculate the sample mean and variance of the log-transformed data
        n = len(log_data)
        sample_mean_log = np.mean(log_data)
        sample_variance_log = np.var(log_data, ddof=1)  # Using unbiased estimator with ddof=1

        # Step 3: Calculate Sichel's function ψ_n
        if sighel == "log":
            psi_n = Stats.sichel_function_log(0.5 * sample_variance_log, n)
        else:
            psi_n = Stats.sichel_function(0.5 * sample_variance_log, n, M)

        # Step 4: Estimate the mean of the log-normal distribution using MVUE
        mvue_mean = np.exp(sample_mean_log) * psi_n

        # Step 5: Calculate the confidence interval for the mean of the log-normal distribution
        t_alpha_2 = stats.t.ppf(1 - (1 - confidence) / 2, df=n-1)
        ci_lower_log = sample_mean_log - t_alpha_2 * np.sqrt(sample_variance_log / n)
        ci_upper_log = sample_mean_log + t_alpha_2 * np.sqrt(sample_variance_log / n)

        mvue_ci_lower = np.exp(ci_lower_log) * psi_n
        mvue_ci_upper = np.exp(ci_upper_log) * psi_n

        return mvue_mean, mvue_ci_lower, mvue_ci_upper

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
    def median_absolute_deviation(data, axis=None, func=None, ignore_nan=False):
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
