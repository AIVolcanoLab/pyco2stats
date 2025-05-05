import numpy as np
import math
import warnings
from scipy.special import hyp0f1
from scipy.stats import t, norm, chi2

class EnvStatsPy:
    """
    Python class exposing the R function elnormAlt with full support
    for different estimation methods and confidence interval (CI) types.
    """

    @staticmethod
    def lognormal_estimator(
            data,
            method="umvue_finney",
            ci=False,
            ci_method="land",
            ci_type="two-sided",
            conf_level=0.95,
            parkin_list=None
        ):
            """
            Estimate the mean and (optionally) confidence interval of a log‐normal distribution.
            
            Parameters
            ----------
            data : array_like
                Positive sample from a log‐normal distribution.
            method : str
                One of "umvue_finney", "sichel", "qmle", "mle", "mme", "mmue".
            ci : bool
                Whether to compute a confidence interval.
            ci_method : str
                One of "land", "normal.approx", "zou", "cox", "parkin", "sichel".
            ci_type : str
                "two-sided", "lower", or "upper".
            conf_level : float
                Confidence level for the interval (e.g. 0.95).
            parkin_list : any
                Optional custom quantiles for Parkin’s method.
            
            Returns
            -------
            dict
            """
            data = np.asarray(data, dtype=float)
            if data.size < 2 or np.any(data <= 0):
                raise ValueError("`data` must contain at least two positive values.")
            
            n = data.size
            log_data = np.log(data)
            mu_hat = float(np.mean(log_data))
            sigma2_hat = float(np.var(log_data, ddof=1))
            sigma2_mle = sigma2_hat * (n-1)/n
            
            # Point and variance estimation
            if method == "finney":
                theta_hat, varhat = EnvStatsPy.umvue_finney_lognormal_estimator(data)
            elif method == "sichel":
                theta_hat, varhat = EnvStatsPy.umvue_sichel_lognormal_estimator(data)
            elif method == "qmle":
                theta_hat = math.exp(mu_hat + 0.5*sigma2_hat)
                varhat = theta_hat**2 * (math.exp(sigma2_hat) - 1)
            elif method == "mle":
                theta_hat = math.exp(mu_hat + 0.5*sigma2_mle)
                varhat = theta_hat**2 * (math.exp(sigma2_mle) - 1)
            elif method == "mme":
                theta_hat = float(np.mean(data))
                varhat = float(np.var(data, ddof=1) / n)
            elif method == "mmue":
                theta_hat = float(np.mean(data))
                varhat = float(np.var(data, ddof=0) / n)
            else:
                raise ValueError(f"Unknown method '{method}'")
            
            # Safe sd calculation
            try:
                sd_hat = math.sqrt(float(varhat))
            except Exception:
                sd_hat = float("nan")
            
            result = {
                "distribution": "Lognormal",
                "sample_size": n,
                "method": method,
                "mean_estimate": theta_hat,
                "sd_estimate": sd_hat
            }
            
            # Confidence interval dispatch
            if ci:
                alpha = 1 - conf_level
                
                if ci_method == "land":
                    ci_limits = EnvStatsPy.ci_lnorm_land(
                        meanlog=mu_hat,
                        sdlog=np.sqrt(sigma2_hat),
                        n=n,
                        ci_type=ci_type,
                        conf_level=conf_level
                    )
                
                elif ci_method in ("normal.approx", "normal.approximation"):
                    # Normal or t-based CI
                    df = n - 1
                    if ci_type in ("two-sided", "two.sided"):
                        q = alpha/2
                    else:
                        q = alpha
                    quantile = t.ppf(1 - q, df)
                    hw = quantile * sd_hat
                    if ci_type in ("two-sided", "two.sided"):
                        lcl, ucl = theta_hat - hw, theta_hat + hw
                    elif ci_type == "lower":
                        lcl, ucl = theta_hat - hw, np.inf
                    else:
                        lcl, ucl = -np.inf, theta_hat + hw
                    ci_limits = {"LCL": lcl, "UCL": ucl}
                
                elif ci_method == "zou":
                    ci_limits = EnvStatsPy.ci_lnorm_zou(
                        meanlog=mu_hat,
                        sdlog=np.sqrt(sigma2_hat),
                        n=n,
                        ci_type=ci_type,
                        conf_level=conf_level
                    )
                
                elif ci_method == "cox":
                    ci_limits = EnvStatsPy.ci_cox(
                        meanlog=mu_hat,
                        s2=sigma2_hat,
                        n=n,
                        ci_type=ci_type,
                        conf_level=conf_level
                    )
                
                elif ci_method == "parkin":
                    ci_limits = EnvStatsPy.ci_parkin(
                        data, ci_type, conf_level, parkin_list
                    )
                
                elif ci_method == "sichel":
                    ci_limits = EnvStatsPy.ci_sichel(
                        mu_hat=mu_hat,
                        sigma2_hat=sigma2_hat,
                        n=n,
                        ci_type=ci_type,
                        conf_level=conf_level
                    )
                
                else:
                    raise ValueError(f"Unknown ci_method '{ci_method}'")
                
                result["confidence_interval"] = ci_limits
            
            return result

    @staticmethod
    def umvue_finney_lognormal_estimator(data):
        """
        UMVUE of the log‑normal mean & variance (Finney’s formula).
        Returns (mean_estimate, variance_estimate).  If data has fewer
        than two points, variance_estimate is NaN.  If data contains
        non‐positive values, returns (NaN, NaN) with a warning.
        """
        data = np.asarray(data)
        n = data.size

        if n == 0:
            return np.nan, np.nan
        if np.any(data <= 0):
            warnings.warn(
                "Data contains non‐positive values; lognormal requires positive data.",
                UserWarning
            )
            return np.nan, np.nan
        if n == 1:
            return float(data[0]), np.nan

        # Log-space moments
        log_data = np.log(data)
        y_bar = log_data.mean()
        s_sq  = log_data.var(ddof=1)

        # Finney’s UMVUE for the mean
        alpha = (n - 1.0) / 2.0
        z     = (n - 1.0)**2 / (4.0 * n) * s_sq
        
        phi = EnvStatsPy.finneys_g(n - 1, s_sq/2)
        umvu_mean = np.exp(y_bar) * phi
    
        # Finney’s UMVUE for the variance (only defined for n>2)
        if n > 2:
            umvu_variance = np.exp(2 * y_bar) * (EnvStatsPy.finneys_g(n - 1, 2 * s_sq) - EnvStatsPy.finneys_g(n - 1, (s_sq * (n - 2))/(n - 1)))
        else:
            umvu_variance = np.nan

        return umvu_mean, umvu_variance



    def finneys_g(m, z, n_terms_inc=10, max_iter=100, tol=None):
        """
        Compute Finney's g_m(z) via iterative summation of series terms.

        Parameters
        ----------
        m : int or array_like of int
            Must be integer(s) >= 1.
        z : float or array_like of float
            Series argument(s).
        n_terms_inc : int, optional
            Initial block size for terms (>= 3).
        max_iter : int, optional
            Maximum number of blocks to sum (>= 1).
        tol : float, optional
            Tolerance for the magnitude of the last term; default is machine epsilon.

        Returns
        -------
        g_values : float or ndarray
            The summed series value(s) of g_m(z).

        Raises
        ------
        ValueError
            If inputs are invalid or series fails to converge.
        """
        # Validate inputs
        tol = tol if tol is not None else np.finfo(float).eps
        m_arr = np.atleast_1d(m).astype(int)
        z_arr = np.atleast_1d(z).astype(float)
        if np.any(m_arr < 1):
            raise ValueError("All values of 'm' must be integers >= 1")
        if n_terms_inc < 3 or int(n_terms_inc) != n_terms_inc:
            raise ValueError("'n_terms_inc' must be an integer >= 3")
        if max_iter < 1 or int(max_iter) != max_iter:
            raise ValueError("'max_iter' must be an integer >= 1")
        if tol < np.finfo(float).eps:
            raise ValueError(f"'tol' must be >= {np.finfo(float).eps}")

        # Broadcast shapes
        m_arr, z_arr = np.broadcast_arrays(m_arr, z_arr)
        result = np.empty_like(z_arr, dtype=float)

        # Helper: compute series terms up to n_terms
        def _series_terms(m_i, z_i, n_terms):
            p = np.arange(2, n_terms)
            num = np.concatenate(([0],  # log(1) = 0
                                  [np.log(m_i) + np.log(abs(z_i))],
                                  2*p*np.log(m_i) + np.log(m_i + 2*p) + p*np.log(abs(z_i))))
            cumsum_m2p = np.cumsum(np.log(m_i + 2*p))
            cumsum_p   = np.cumsum(np.log(p))
            denom = np.concatenate(([0],  # log(1) = 0
                                    [np.log(m_i + 1)],
                                    np.log(m_i) + np.log(m_i + 2) + cumsum_m2p + p*np.log(m_i + 1) + cumsum_p))
            terms = np.exp(num - denom)
            if z_i < 0:
                signs = (-1)**np.arange(len(terms))
                terms *= signs
            return terms

        # Main loop: iterate blocks until convergence
        for idx, (m_i, z_i) in enumerate(zip(m_arr, z_arr)):
            converged = False
            for block in range(1, max_iter+1):
                n_terms = n_terms_inc * block
                terms = _series_terms(m_i, z_i, n_terms)
                if abs(terms[-1]) <= tol:
                    result[idx] = terms.sum()
                    converged = True
                    break
            if not converged:
                raise ValueError(f"Series failed to converge for m={m_i}, z={z_i}")
        return result if result.shape != () else float(result)



    #------------------------------------------------------------------------------  
    # Example Sichel Psi-factor table: extend with full data as needed  
    # Stored as list of tuples: (n, V, psi_lower, psi_upper)  
    psi_table = [  
        (3,  0.20, 0.58, 1.87),  
        (3,  0.40, 0.49, 2.29),  
        (5,  0.20, 0.67, 1.73),  
        (5,  0.40, 0.55, 2.08),  
        (10, 0.20, 0.75, 1.59),  
        (10, 0.40, 0.64, 1.82),  
        # ... add the full table here ...  
    ]

    def lookup_psi(p, V, n, psi_table=psi_table):
        """
        Lookup (or nearest‐neighbor) Sichel Psi‐factor for tail probability p, statistic V, and sample size n.
        p: tail probability (e.g., alpha/2 or 1-alpha/2)
        V: Sichel V statistic = sqrt(sigma^2 / n)
        n: sample size
        """
        # Filter entries for given n
        entries = [entry for entry in psi_table if entry[0] == n]
        if not entries:
            raise ValueError(f"No Psi-factors available for n={n}")
        # Try exact match on V
        for _, v_val, psi_l, psi_u in entries:
            if np.isclose(v_val, V):
                return psi_l if p < 0.5 else psi_u
        # Nearest V fallback
        _, _, psi_l, psi_u = min(entries, key=lambda e: abs(e[1] - V))
        return psi_l if p < 0.5 else psi_u

    def umvue_sichel_lognormal_estimator(data):
        """
        Compute Sichel's unbiased estimator of the lognormal mean and the fitted lognormal variance.

        Parameters
        ----------
        data : array_like
            Positive-valued sample from a lognormal distribution.

        Returns
        -------
        dict
            {
              "mean_estimate": float,
              "variance_estimate": float,
            }
        """
        data = np.asarray(data, dtype=float)
        if np.any(data <= 0):
            raise ValueError("All observations must be positive for lognormal fitting.")
        
        log_data = np.log(data)
        n = len(log_data)
        
        # Sample log-space estimates
        hat_mu     = np.mean(log_data)
        hat_sigma2 = np.var(log_data, ddof=1)
        
        # Sichel point estimate
        z1 = (n-1)/2
        z2 = hat_sigma2*(n-1)/4
        gamma_n = hyp0f1(z1, z2)
        mean_est  = np.exp(hat_mu) * gamma_n
        
        # Fitted distribution variance (plug-in formula)
        variance_est = (np.exp(hat_sigma2) - 1) * np.exp(2*hat_mu + hat_sigma2)

        
        return mean_est, variance_est

    def umvue_sichel_lognormal_estimator_old(data):
            """
            Compute Sichel's unbiased estimator of the lognormal mean and the fitted lognormal variance.

            Parameters
            ----------
            data : array_like
                Positive-valued sample from a lognormal distribution.

            Returns
            -------
            dict
                {
                  "mean_estimate": float,
                  "variance_estimate": float,
                }
            """
            data = np.asarray(data, dtype=float)
            if np.any(data <= 0):
                raise ValueError("All observations must be positive for lognormal fitting.")
            
            log_data = np.log(data)
            n = len(log_data)
            
            # Sample log-space estimates
            hat_mu     = np.mean(log_data)
            hat_sigma2 = np.var(log_data, ddof=1)
            
            # Sichel point estimate
            z = (n - 1) * hat_sigma2 / 2
            gamma_num = hyp0f1(n/2,     z)
            gamma_den = hyp0f1((n+1)/2, z)
            gamma_n   = gamma_num / gamma_den
            mean_est  = np.exp(hat_mu) * gamma_n
            
            # Fitted distribution variance (plug-in formula)
            variance_est = (np.exp(hat_sigma2) - 1) * np.exp(2*hat_mu + hat_sigma2)

            
            return mean_est, variance_est
        

    @staticmethod
    def ci_lnorm_land(meanlog, sdlog, n, ci_type, conf_level):
        ci = EnvStatsPy.ci_land(0.5, meanlog, sdlog**2, n, n - 1, n, ci_type, conf_level)
        return {"LCL": np.exp(ci["LCL"]), "UCL": np.exp(ci["UCL"]) }

    @staticmethod
    def ci_land(lambda_, mu_hat, sig_sq_hat, n, nu, gamma_sq, ci_type, conf_level):
        k = (nu + 1) / (2 * lambda_ * gamma_sq)
        S = np.sqrt((2 * lambda_ * sig_sq_hat) / k)
        alpha = 1 - conf_level

        if ci_type == "two-sided":
            lcl = mu_hat + lambda_ * sig_sq_hat + (k * S / np.sqrt(nu)) * t.ppf(alpha/2, nu)
            ucl = mu_hat + lambda_ * sig_sq_hat + (k * S / np.sqrt(nu)) * t.ppf(1 - alpha/2, nu)
        elif ci_type == "lower":
            lcl = mu_hat + lambda_ * sig_sq_hat + (k * S / np.sqrt(nu)) * t.ppf(alpha, nu)
            ucl = np.inf
        else:
            lcl = -np.inf
            ucl = mu_hat + lambda_ * sig_sq_hat + (k * S / np.sqrt(nu)) * t.ppf(1 - alpha, nu)

        return {"LCL": lcl, "UCL": ucl}

    def ci_sichel(mu_hat, sigma2_hat, n, ci_type="two-sided", conf_level=0.95):
        """
        Compute Sichel's confidence interval for the lognormal mean.

        Parameters
        ----------
        mu_hat : float
            Sample mean of log-data, Ȳ.
        sigma2_hat : float
            Unbiased sample variance of log-data, S_Y^2.
        n : int
            Sample size.
        ci_type : str
            "two-sided", "lower", or "upper".
        conf_level : float
            Confidence level between 0 and 1.

        Returns
        -------
        dict
            {"LCL": lower_conf_limit, "UCL": upper_conf_limit}.
        """
        # Compute point estimate
        V = np.sqrt(sigma2_hat / n)
        z = (n - 1) * sigma2_hat / 2
        gamma_n = hyp0f1(n/2, z) / hyp0f1((n+1)/2, z)
        mean_est = np.exp(mu_hat) * gamma_n

        alpha = 1 - conf_level
        # Determine tail probabilities
        if ci_type == "two-sided":
            p_lower, p_upper = alpha/2, 1 - alpha/2
        elif ci_type == "lower":
            p_lower, p_upper = alpha, None
        else:  # upper only
            p_lower, p_upper = None, 1 - alpha

        # Lookup psi-factors and compute limits
        lcl = mean_est * EnvStatsPy.lookup_psi(p_lower, V, n) if p_lower is not None else -np.inf
        ucl = mean_est * EnvStatsPy.lookup_psi(p_upper, V, n) if p_upper is not None else np.inf

        return {"LCL": lcl, "UCL": ucl}


        @staticmethod
        def ci_normal_approx_old(mean, sd, n, ci_type, conf_level):
            alpha = 1 - conf_level
            se = sd / np.sqrt(n)

            if ci_type == "two-sided":
                z_val = norm.ppf(1 - alpha/2)
                return {"LCL": mean - z_val*se, "UCL": mean + z_val*se}
            elif ci_type == "lower":
                z_val = norm.ppf(1 - alpha)
                return {"LCL": mean - z_val*se, "UCL": np.inf}
            else:
                z_val = norm.ppf(1 - alpha)
                return {"LCL": -np.inf, "UCL": mean + z_val*se}

        @staticmethod
        def ci_normal_approx(
            theta_hat,
            sd_theta_hat,
            n,
            df=None,
            ci_type="two-sided",
            alpha=0.05,
            lb=-np.inf,
            ub=np.inf,
            test_statistic="z"
        ):
            """
            Compute a confidence interval for theta using normal approximation.

            Parameters
            ----------
            theta_hat : float
                Point estimate of the parameter theta.
            sd_theta_hat : float
                Estimated standard deviation of theta_hat.
            n : int
                Sample size.
            df : int or None, optional
                Degrees of freedom for the t distribution; required if test_statistic='t'.
            ci_type : {'two-sided', 'lower', 'upper'}, optional
                Type of interval.
            alpha : float, optional
                Significance level (1 - confidence level).
            lb : float, optional
                Lower bound for truncation (default -inf).
            ub : float, optional
                Upper bound for truncation (default +inf).
            test_statistic : {'t', 'z'}, optional
                Use Student's t ('t') or normal ('z') quantiles.

            Returns
            -------
            dict
                {'LCL': lower_confidence_limit, 'UCL': upper_confidence_limit}
            """
            test_statistic = test_statistic.lower()
            if test_statistic == "t" and df is None:
                raise ValueError("When test_statistic='t', you must supply df")
            # Determine half-width
            if ci_type in ("two-sided", "two.sided"):
                q = alpha / 2
            else:
                q = alpha
            if test_statistic == "t":
                quantile = t.ppf(1 - q, df)
            else:
                quantile = norm.ppf(1 - q)
            hw = quantile * sd_theta_hat

            if ci_type in ("two-sided", "two.sided"):
                lcl = max(lb, theta_hat - hw)
                ucl = min(ub, theta_hat + hw)
            elif ci_type == "lower":
                lcl = max(lb, theta_hat - hw)
                ucl = ub
            elif ci_type == "upper":
                lcl = lb
                ucl = min(ub, theta_hat + hw)
            else:
                raise ValueError("ci_type must be 'two-sided', 'lower', or 'upper'")

            return {"LCL": lcl, "UCL": ucl}

    @staticmethod
    def ci_lnorm_zou(meanlog, sdlog, n, ci_type, conf_level):
        alpha = 1 - conf_level
        pivot = meanlog + sdlog**2 / 2
        se_mean = sdlog / np.sqrt(n)

        if ci_type == "two-sided":
            z = norm.ppf(1 - alpha/2)
        else:
            z = norm.ppf(1 - alpha)

        chi2_l, chi2_u = chi2.ppf(alpha/2, n-1), chi2.ppf(1-alpha/2, n-1)
        theta2_l = (n-1) * sdlog**2 / chi2_u
        theta2_u = (n-1) * sdlog**2 / chi2_l

        if ci_type == "two-sided":
            lcl = np.exp(pivot - np.sqrt((se_mean*z)**2 + (sdlog**2/2 - theta2_l/2)**2))
            ucl = np.exp(pivot + np.sqrt((se_mean*z)**2 + (theta2_u/2 - sdlog**2/2)**2))
        elif ci_type == "lower":
            lcl = np.exp(pivot - np.sqrt((se_mean*z)**2 + (sdlog**2/2 - theta2_l/2)**2))
            ucl = np.inf
        else:
            lcl = -np.inf
            ucl = np.exp(pivot + np.sqrt((se_mean*z)**2 + (theta2_u/2 - sdlog**2/2)**2))

        return {"LCL": lcl, "UCL": ucl}

    @staticmethod
    def ci_cox(meanlog, s2, n, ci_type, conf_level):
        beta_hat = meanlog + s2 / 2
        sd_beta_hat = np.sqrt(s2/n + (s2**2)/(2*(n - 1))) # sd_beta_hat = np.sqrt(s2/n + (s2**2)/(2*(n + 1))) Original
        ci = EnvStatsPy.ci_normal_approx(beta_hat, sd_beta_hat, n, ci_type, conf_level, test_statistic="t")
        return {"LCL": np.exp(ci["LCL"]), "UCL": np.exp(ci["UCL"]) }

    @staticmethod
    def ci_parkin(data, ci_type, conf_level):
        data = np.sort(data)
        n = len(data)
        p = 0.5
        rank = int(np.ceil(p*n)) - 1
        lcl, ucl = data[max(0,rank)], data[min(n-1,rank)]

        if ci_type == "two-sided":
            return {"LCL": lcl, "UCL": ucl}
        elif ci_type == "lower":
            return {"LCL": lcl, "UCL": np.inf}
        else:
            return {"LCL": -np.inf, "UCL": ucl}
