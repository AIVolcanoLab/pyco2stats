import warnings
import numpy as np
import math
from scipy.optimize import minimize
from scipy.special import hyp0f1
from scipy.stats import t, norm, chi2, nct




class EnvStatsPy:
    """
    Python implementation of R's elnormAlt for lognormal mean estimation 
    with various estimators and confidence intervals.
    """

    @staticmethod
    def lognormal_estimator(
        X_lognorm_data,
        method="umvue_finney",
        ci=False,
        ci_method="zou",
        ci_type="two-sided",
        conf_level=0.95,
        parkin_list=None
    ):
        X_lognorm_data = np.asarray(X_lognorm_data, dtype=float)
        if X_lognorm_data.size < 2 or np.any(X_lognorm_data <= 0):
            raise ValueError("`data` must contain at least two positive values.")

        n = X_lognorm_data.size
        Y_norm_data = np.log(X_lognorm_data)
        Y_mu_hat = float(np.mean(Y_norm_data))
        Y_sigma2_hat = float(np.var(Y_norm_data, ddof=1))
        Y_sigma2_mle = Y_sigma2_hat * (n - 1) / n

        # Point estimation
        if method == "umvue_finney":
            X_theta_hat, X_var_hat = EnvStatsPy.umvue_finney_lognormal_estimator(X_lognorm_data)
        elif method == "umvue_sichel":
            X_theta_hat, X_var_hat = EnvStatsPy.umvue_sichel_lognormal_estimator(X_lognorm_data)
        elif method == "qmle":
            X_theta_hat = math.exp(Y_mu_hat + 0.5 * Y_sigma2_hat)
            se2 = (np.exp(Y_sigma2_hat) - 1) * np.exp(2 * Y_mu_hat + Y_sigma2_hat)
            X_var_hat = se2 / n
        elif method == "mle":
            X_theta_hat = math.exp(Y_mu_hat + 0.5 * Y_sigma2_mle)
            se2 = (np.exp(Y_sigma2_mle) - 1) * np.exp(2 * Y_mu_hat + Y_sigma2_mle)
            X_var_hat = se2 / n
        elif method == "mme":
            X_theta_hat = float(np.mean(X_lognorm_data))
            X_var_hat = float(np.var(X_lognorm_data, ddof=1) / n)
        elif method == "mmue":
            X_theta_hat = float(np.mean(X_lognorm_data))
            X_var_hat = float(np.var(X_lognorm_data, ddof=0) / n)
        else:
            raise ValueError(f"Unknown method '{method}'")

        # Always compute SE
        if np.isnan(X_var_hat) or X_var_hat < 0:
            X_sd_hat = float("nan")
            warnings.warn(f"Standard deviation estimate not available for method='{method}'.", UserWarning)
        else:
            X_sd_hat = math.sqrt(X_var_hat)

        result = {
            "sample_size": n,
            "method": method,
            "mean_estimate": X_theta_hat,
            "sd_estimate": X_sd_hat
        }

        # Confidence intervals
        if ci:
            if ci_method == "land":
                try:
                    ci_limits = EnvStatsPy.ci_lnorm_land(
                        mu_hat=Y_mu_hat + 0.5 * Y_sigma2_hat,
                        sigma2_hat=Y_sigma2_hat,
                        n=n,
                        ci_type=ci_type,
                        conf_level=conf_level
                    )
                except RuntimeError as e:
                    warnings.warn(f"Land CI failed due to optimization error: {e}. Falling back to Zou method.", UserWarning)
                    ci_limits = EnvStatsPy.ci_lnorm_zou(
                        mu_hat=Y_mu_hat,  
                        sigma2_hat=Y_sigma2_hat,
                        n=n,
                        ci_type=ci_type,
                        conf_level=conf_level
                    )
            elif ci_method == "normal_approx":
                if method in ["mme", "mmue"]:
                    ci_limits = EnvStatsPy.ci_standard_approx(
                        mu_hat=X_theta_hat,
                        sigma2_hat=X_sd_hat**2,
                        n=n,
                        test_statistic="z",
                        ci_type=ci_type,
                        conf_level=conf_level
                    )
                else:
                    warnings.warn(
                        f"'normal_approx' CI is not recommended with method='{method}'. "
                        "It assumes normality of the arithmetic mean, which may not hold for lognormal-transformed estimates.",
                        UserWarning
                    )
                    ci_limits = {"LCL": float("nan"), "UCL": float("nan")}
            elif ci_method == "zou":
                ci_limits = EnvStatsPy.ci_lnorm_zou(
                    mu_hat=Y_mu_hat,  
                    sigma2_hat=Y_sigma2_hat,
                    n=n,
                    ci_type=ci_type,
                    conf_level=conf_level
                )
            elif ci_method == "cox":
                if method in ["umvue_finney", "umvue_sichel"]:
                    warnings.warn(
                        "Using Cox CI with a UMVUE estimator: SE estimate returned is the UMVUE-based SE, not Cox-specific.",
                        UserWarning
                    )
                ci_limits = EnvStatsPy.ci_cox(
                    mu_hat=Y_mu_hat,
                    sigma2_hat=Y_sigma2_hat,
                    n=n,
                    ci_type=ci_type,
                    conf_level=conf_level
                )
            elif ci_method == "parkin":
                ci_limits = EnvStatsPy.ci_parkin(X_lognorm_data, ci_type, conf_level, parkin_list)
            elif ci_method == "sichel":
                ci_limits = EnvStatsPy.ci_sichel(
                    mu_hat=Y_mu_hat,
                    sigma2_hat=Y_sigma2_hat,
                    n=n,
                    ci_type=ci_type,
                    conf_level=conf_level
                )
            else:
                raise ValueError(f"Unknown ci_method '{ci_method}'")

            result.update(ci_limits)

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



    @staticmethod
    def umvue_sichel_lognormal_estimator(X_lognorm_data):
        X_lognorm_data = np.asarray(X_lognorm_data, dtype=float)
        if np.any(X_lognorm_data <= 0):
            raise ValueError("All observations must be positive.")

        log_data = np.log(X_lognorm_data)
        n = len(log_data)
        hat_mu = np.mean(log_data)
        hat_sigma2 = np.var(log_data, ddof=1)

        z1 = (n - 1) / 2
        z2 = hat_sigma2 * (n - 1) / 4
        try:
            gamma_n = hyp0f1(z1, z2)
        except Exception as e:
            warnings.warn(f"Sichel estimator hyp0f1 failed: {e}", RuntimeWarning)
            return np.nan, np.nan

        mean_est = math.exp(hat_mu) * gamma_n
        variance_est = (math.exp(hat_sigma2) - 1) * math.exp(2 * hat_mu + hat_sigma2)

        return mean_est, variance_est


    @staticmethod
    def finneys_g(m, z, n_terms_inc=10, max_iter=100, tol=None):
        tol = tol if tol is not None else np.finfo(float).eps
        m_arr = np.atleast_1d(m).astype(int)
        z_arr = np.atleast_1d(z).astype(float)
        result = np.full_like(z_arr, np.nan, dtype=float)

        def _terms(m_i, z_i, n_terms):
            p = np.arange(2, n_terms)
            num = np.concatenate(([0], [math.log(m_i) + math.log(abs(z_i))],
                                   2*p*math.log(m_i) + np.log(m_i + 2*p) + p*math.log(abs(z_i))))
            cumsum_m2p = np.cumsum(np.log(m_i + 2*p))
            cumsum_p = np.cumsum(np.log(p))
            denom = np.concatenate(([0], [math.log(m_i + 1)],
                                     math.log(m_i) + math.log(m_i + 2) + cumsum_m2p
                                     + p*math.log(m_i + 1) + cumsum_p))
            terms = np.exp(num - denom)
            if z_i < 0:
                terms *= (-1)**np.arange(len(terms))
            return terms

        for idx, (m_i, z_i) in enumerate(zip(m_arr, z_arr)):
            converged = False
            for block in range(1, max_iter+1):
                n_terms = n_terms_inc * block
                try:
                    terms = _terms(m_i, z_i, n_terms)
                    if not np.isfinite(terms).all():
                        raise ValueError("Non-finite terms in series.")
                    if abs(terms[-1]) <= tol:
                        result[idx] = terms.sum()
                        converged = True
                        break
                except Exception as e:
                    msg = f"finneys_g failed at index {idx}: m={m_i}, z={z_i}, reason: {e}"
                    warnings.warn(msg, RuntimeWarning)
                    break
            if not converged:
                msg = f"finneys_g did not converge at index {idx}: m={m_i}, z={z_i}"
                warnings.warn(msg, RuntimeWarning)

        return float(result) if result.size == 1 else result

    # Example Psi-factor table; extend as needed
    psi_table = [
        (3, 0.20, 0.58, 1.87),
        (3, 0.40, 0.49, 2.29),
        (5, 0.20, 0.67, 1.73),
        (5, 0.40, 0.55, 2.08),
        (10,0.20, 0.75, 1.59),
        (10,0.40, 0.64, 1.82),
    ]

    @staticmethod
    def lookup_psi(p, V, n, psi_table=psi_table):
        entries = [e for e in psi_table if e[0] == n]
        if not entries:
            raise ValueError(f"No Psi-factors for n={n}")
        for _, v_val, psi_l, psi_u in entries:
            if math.isclose(v_val, V):
                return psi_l if p < 0.5 else psi_u
        _, _, psi_l, psi_u = min(entries, key=lambda e: abs(e[1] - V))
        return psi_l if p < 0.5 else psi_u

    import math


    def ci_lnorm_land(mu_hat, sigma2_hat, n, ci_type="two-sided", conf_level=0.95):
        """
        Land's CI for lognormal mean matching R implementations ci.lnorm.land, lands.C, ci.land.
        """
        # Input validation 
        if sigma2_hat <= 0:
            raise ValueError("sigma2_hat must be positive")
        if n < 2 or int(n) != n:
            raise ValueError("n must be integer >= 2")
        if not (0.5 <= conf_level < 1):
            raise ValueError("conf_level must be in [0.5, 1)")
        ci_type = ci_type.lower()
        if ci_type not in ("two-sided", "lower", "upper"):
            raise ValueError("ci_type must be 'two-sided', 'lower', or 'upper'")

        # Land method parameters
        lam = 0.5
        nu = n - 1
        gamma_sq = n
        k = (nu + 1) / (2 * lam * gamma_sq)
        S = math.sqrt((2 * lam * sigma2_hat) / k)

        # Helper function: non-central t quantile
        def qlands_t(p, nu, zeta):
            return nct.ppf(p, df=nu, nc=zeta)

        # lands.C equivalent in Python
        def lands_C(S, nu, alpha):
            """
            Improved Land's C function with stabilized optimization and quantile logic.
            """

            if S <= 0:
                raise ValueError("'S' must be positive.")
            if nu < 2:
                raise ValueError("'nu' must be at least 2.")
            if not (0 < alpha < 1):
                raise ValueError("'alpha' must be between 0 and 1.")

            def qlands_t(p, nu, zeta):
                """Stable non-central t-quantile"""
                try:
                    # Clip large zeta values to avoid nct.ppf instability
                    if abs(zeta) > 100:
                        zeta = 100 * math.copysign(1, zeta)
                    qt = nct.ppf(p, df=nu, nc=zeta)
                    if not np.isfinite(qt):
                        raise ValueError("Non-finite result")
                    return qt
                except Exception:
                    # Fallback to standard t approximation
                    warnings.warn("Falling back to central t-distribution for quantile")
                    return nct.ppf(p, df=nu, nc=0)

            def objective(m):
                T_m = (math.sqrt(nu + 1) * ((-S**2) / 2 - m)) / S
                zeta_m = (-S * math.sqrt(nu + T_m**2)) / (2 * math.sqrt(nu + 1))
                try:
                    quantile = qlands_t(alpha, nu, zeta_m)
                except Exception:
                    quantile = float('inf')
                return (T_m - quantile) ** 2

            # Initial guess (same as R)
            T0 = (-math.sqrt(nu + 1) * S) / 2
            zeta0 = (-S * math.sqrt(nu + T0**2)) / (2 * math.sqrt(nu + 1))
            try:
                T0 = qlands_t(alpha, nu, zeta0)
            except Exception:
                T0 = -T0
            m0 = (-S**2) / 2 - (S * T0) / math.sqrt(nu + 1)

            # Use bounded optimization like R
            bounds = [(-10 * abs(m0), 10 * abs(m0))]

            res = minimize(
                objective,
                x0=[m0],
                method='L-BFGS-B',
                bounds=bounds,
                options={'ftol': 1e-12, 'gtol': 1e-12, 'maxiter': 20000}
            )

            if not res.success:
                raise RuntimeError(f"Optimization failed: {res.message}")

            m_opt = res.x.item()
            return (m_opt * math.sqrt(nu)) / S

        # CI calculations matching R logic exactly
        alpha = 1 - conf_level
        if ci_type == "two-sided":
            psi_low = lands_C(S, nu, alpha / 2)
            psi_high = lands_C(S, nu, 1 - alpha / 2)
            lcl = mu_hat + lam * sigma2_hat + (k * S / math.sqrt(nu)) * psi_low
            ucl = mu_hat + lam * sigma2_hat + (k * S / math.sqrt(nu)) * psi_high
        elif ci_type == "lower":
            psi_low = lands_C(S, nu, alpha)
            lcl = mu_hat + lam * sigma2_hat + (k * S / math.sqrt(nu)) * psi_low
            ucl = math.inf
        else:  # upper
            psi_high = lands_C(S, nu, conf_level)
            lcl = -math.inf
            ucl = mu_hat + lam * sigma2_hat + (k * S / math.sqrt(nu)) * psi_high

        # Exponentiate to original-scale mean CI
        lcl_exp, ucl_exp = sorted([math.exp(lcl), math.exp(ucl)])
        return {"LCL": lcl_exp, "UCL": ucl_exp}




    @staticmethod
    def ci_standard_approx(
        mu_hat, sigma2_hat, n,
        df=None, ci_type="two-sided", conf_level=0.95,
        lb=-math.inf, ub=math.inf, test_statistic="z"
    ):
        sd_hat = np.sqrt(sigma2_hat)
        alpha = 1 - conf_level
        test_statistic = test_statistic.lower()
        df = df if df is not None else (n-1 if test_statistic=="t" else None)
        if test_statistic == "t" and df is None:
            raise ValueError("df required for t-interval")
        q_lower, q_upper = {
            "two-sided": (alpha/2, 1-alpha/2),
            "lower": (alpha, 1.0),
            "upper": (0.0, 1-alpha)
        }.get(ci_type, (None,None))
        quant = (lambda p: t.ppf(p, df)) if test_statistic=="t" else (lambda p: norm.ppf(p))
        lcl = mu_hat - quant(1-q_lower)*sd_hat if q_lower>0 else -math.inf
        ucl = mu_hat + quant(q_upper)*sd_hat if q_upper<1 else math.inf
        return {"LCL": max(lb, lcl), "UCL": min(ub, ucl)}

    @staticmethod
    def ci_lnorm_zou(mu_hat, sigma2_hat, n, ci_type, conf_level):
        """
        Exact port of R's ci.lnorm.zou() for confidence intervals on the lognormal mean.
        """
        alpha = 1 - conf_level
        sdlog = np.sqrt(sigma2_hat)
        theta2_hat = sigma2_hat / 2
        pivot = mu_hat + theta2_hat

        # Mean CI component (z-distribution)
        z = norm.ppf(1 - alpha / 2) if ci_type == "two-sided" else norm.ppf(1 - alpha)
        se_meanlog = sdlog / np.sqrt(n)
        mean_LCL = mu_hat - z * se_meanlog
        mean_UCL = mu_hat + z * se_meanlog

        # Variance CI component (chi-squared)
        df = n - 1
        chi2_L = chi2.ppf(alpha / 2, df) if ci_type == "two-sided" else chi2.ppf(alpha, df)
        chi2_U = chi2.ppf(1 - alpha / 2, df) if ci_type == "two-sided" else chi2.ppf(1 - alpha, df)

        var_LCL = (sdlog**2 * df / chi2_U) / 2
        var_UCL = (sdlog**2 * df / chi2_L) / 2

        if ci_type == "two-sided":
            dL = (mu_hat - mean_LCL)**2 + (theta2_hat - var_LCL)**2
            dU = (mean_UCL - mu_hat)**2 + (var_UCL - theta2_hat)**2
            LCL = math.exp(pivot - math.sqrt(dL))
            UCL = math.exp(pivot + math.sqrt(dU))
        elif ci_type == "lower":
            dL = (mu_hat - mean_LCL)**2 + (theta2_hat - var_LCL)**2
            LCL = math.exp(pivot - math.sqrt(dL))
            UCL = math.inf
        else:  # upper
            dU = (mean_UCL - mu_hat)**2 + (var_UCL - theta2_hat)**2
            LCL = -math.inf
            UCL = math.exp(pivot + math.sqrt(dU))

        return {"LCL": LCL, "UCL": UCL}


    @staticmethod
    def ci_cox(mu_hat, sigma2_hat, n, ci_type="two-sided", conf_level=0.95):
        """
        Cox method for confidence intervals on the lognormal mean.
        Matches the logic of EnvStats::elnormAlt(ci.method="cox") in R.
        """
        alpha = 1 - conf_level
        df = n - 1
        ci_type = ci_type.lower()

        # Point estimate of mean on log scale plus half variance
        beta_hat = mu_hat + (sigma2_hat / 2)

        # Cox standard error formula
        se_beta_hat = math.sqrt(sigma2_hat / n + (sigma2_hat ** 2) / (2 * (n + 1)))

        # z critical value
        z = norm.ppf(1 - alpha / 2) if ci_type == "two-sided" else norm.ppf(1 - alpha)

        if ci_type == "two-sided":
            lcl = beta_hat - z * se_beta_hat
            ucl = beta_hat + z * se_beta_hat
        elif ci_type == "lower":
            lcl = beta_hat - z * se_beta_hat
            ucl = math.inf
        elif ci_type == "upper":
            lcl = -math.inf
            ucl = beta_hat + z * se_beta_hat
        else:
            raise ValueError("ci_type must be 'two-sided', 'lower', or 'upper'")

        # Transform back to original scale
        lcl_exp, ucl_exp = sorted([math.exp(lcl), math.exp(ucl)])
        return {"LCL": lcl_exp, "UCL": ucl_exp}

    @staticmethod
    def ci_parkin(data, ci_type, conf_level, parkin_list=None):
        data = np.sort(np.asarray(data))
        n = len(data)
        rank = int(np.ceil(0.5*n)) - 1
        lcl, ucl = data[max(0, rank)], data[min(n-1, rank)]
        if ci_type == "two-sided":
            return {"LCL": lcl, "UCL": ucl}
        elif ci_type == "lower":
            return {"LCL": lcl, "UCL": math.inf}
        else:
            return {"LCL": -math.inf, "UCL": ucl}

