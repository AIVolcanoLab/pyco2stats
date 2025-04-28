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
    def elnormAlt(data, method="mvue", ci=False, ci_method="land", ci_type="two-sided", conf_level=0.95, parkin_list=None):
        """
        Estimate parameters of a Lognormal distribution.
        """
        data = np.asarray(data)
        if len(data) < 2 or np.any(data <= 0):
            raise ValueError("'data' must have at least 2 positive values.")

        n = len(data)
        log_data = np.log(data)
        meanlog = np.mean(log_data)
        s2 = np.var(log_data, ddof=1)
        s2_mle = s2 * (n - 1) / n

        if method == "mvue":
            muhat, varhat = EnvStatsPy.umvu_lognormal_mean_variance(data)
            sdhat = np.sqrt(varhat) if not np.isnan(varhat) else np.nan
        elif method == "qmle":
            muhat = np.exp(meanlog + s2 / 2)
            sdhat = muhat * np.sqrt(np.exp(s2) - 1)
        elif method == "mle":
            muhat = np.exp(meanlog + s2_mle / 2)
            sdhat = muhat * np.sqrt(np.exp(s2_mle) - 1)
        elif method in ["mme", "mmue"]:
            muhat = np.mean(data)
            if method == "mme":
                sdhat = np.std(data, ddof=0)
            else:
                sdhat = np.std(data, ddof=1)
        else:
            raise NotImplementedError(f"Method '{method}' not implemented.")

        result = {
            "distribution": "Lognormal",
            "sample_size": n,
            "mean_estimate": muhat,
            "sd_estimate": sdhat,
            "method": method
        }

        if ci:
            if ci_method == "land":
                ci_result = EnvStatsPy.ci_lnorm_land(meanlog, np.sqrt(s2), n, ci_type, conf_level)
            elif ci_method == "normal.approx":
                ci_result = EnvStatsPy.ci_normal_approx(muhat, sdhat, n, ci_type, conf_level)
            elif ci_method == "zou":
                ci_result = EnvStatsPy.ci_lnorm_zou(meanlog, np.sqrt(s2), n, ci_type, conf_level)
            elif ci_method == "cox":
                ci_result = EnvStatsPy.ci_cox(meanlog, s2, n, ci_type, conf_level)
            elif ci_method == "parkin":
                ci_result = EnvStatsPy.ci_parkin(data, ci_type, conf_level)
            else:
                raise ValueError(f"Unknown ci_method '{ci_method}'.")

            result["confidence_interval"] = ci_result

        return result

    @staticmethod
    def umvu_lognormal_mean_variance(data, tol=1e-9, max_iter=1000):
        data = np.asarray(data)
        n = len(data)

        if np.any(data <= 0):
            warnings.warn("Data contains non-positive values. Lognormal distribution requires positive data.", UserWarning)
            return np.nan, np.nan

        if n < 1:
            return np.nan, np.nan
        if n == 1:
            return float(data[0]), np.nan

        log_data = np.log(data)
        y_bar = np.mean(log_data)
        s_sq = np.var(log_data, ddof=1)

        alpha_mean = (n - 1.0) / 2.0
        z_mean = (n - 1.0)**2 / (4.0 * n) * s_sq

        try:
            phi_mean_0F1 = hyp0f1(alpha_mean, z_mean)
        except Exception as e:
            warnings.warn(f"Error computing hyp0f1 for mean: {e}", RuntimeWarning)
            phi_mean_0F1 = np.nan

        if not np.isnan(phi_mean_0F1):
            umvu_mean = np.exp(y_bar) * phi_mean_0F1
        else:
            umvu_mean = np.nan

        umvu_variance = np.nan
        if n > 2:
            alpha_var = (n - 1.0) / 2.0
            z_var1 = (n - 1.0)**2 / n * s_sq
            z_var2 = (n - 1.0) * (n - 2.0) / (2.0 * n) * s_sq

            try:
                phi_var1_0F1 = hyp0f1(alpha_var, z_var1)
                phi_var2_0F1 = hyp0f1(alpha_var, z_var2)
            except Exception as e:
                warnings.warn(f"Error computing hyp0f1 for variance: {e}", RuntimeWarning)
                phi_var1_0F1 = np.nan
                phi_var2_0F1 = np.nan

            if not np.isnan(phi_var1_0F1) and not np.isnan(phi_var2_0F1):
                umvu_variance = np.exp(2 * y_bar) * (phi_var1_0F1 - phi_var2_0F1)

        return umvu_mean, umvu_variance

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

    @staticmethod
    def ci_normal_approx(mean, sd, n, ci_type, conf_level):
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
        sd_beta_hat = np.sqrt(s2/n + (s2**2)/(2*(n+1)))
        ci = EnvStatsPy.ci_normal_approx(beta_hat, sd_beta_hat, n, ci_type, conf_level)
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
