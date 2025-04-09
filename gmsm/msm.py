import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import stats

from gmsm.model import (
    msm_likelihood2, msm_ll2, msm_predict, msm_parameter_check,
    msm_std_err
)
from gmsm.likelihood import msm_smooth
from gmsm.utils import msm_clustermat, msm_marginals

class MSM:
    def __init__(self, ret, kbar=1, n_vol=252, para0=None, nw_lag=0):
        """
        Initialize and fit the MSM model.

        :param ret : array-like
            Vector of returns
        :param kbar : int, optional
            Number of frequency components, default is 1
        :param n_vol : int, optional
            Number of trading days in a year, default is 252
        :param para0 : list, optional
            Initial parameter values [m0, b, gammak, sigma]
        :param nw_lag : int, optional
            Number of lags for Newey-West standard errors, default is 0
        """
        params = msm_parameter_check(ret, kbar, para0)

        self.ret = params["dat"]
        self.kbar = params["kbar"]
        self.n_vol = n_vol
        self.nw_lag = nw_lag
        self.para0 = params["start_value"]
        self.lb = params["lb"]
        self.ub = params["ub"]

        # Fit the model
        self._fit()

    def _fit(self):
        # Set up bounds for optimizer
        bounds = list(zip(self.lb, self.ub))

        # Run optimization
        opt_result = minimize(
            msm_ll2,
            self.para0,
            args=(self.kbar, self.ret, self.n_vol),
            method='L-BFGS-B',
            bounds=bounds
        )

        self.para = opt_result.x

        msm_estimate = msm_likelihood2(self.para, self.kbar, self.ret, self.n_vol)

        self.se = msm_std_err(self.para, self.kbar, self.ret, self.n_vol, self.nw_lag)

        coef = self.para.copy()
        if self.kbar == 1:
            coef[1] = np.nan

        coef[3] = coef[3] / np.sqrt(self.n_vol)
        self.se[3] = self.se[3] / np.sqrt(self.n_vol)

        self.coef_names = ["m0", "b", "gammak", "sigma"]

        self.results = {
            "LL": msm_estimate["LL"],
            "LLs": msm_estimate["LLs"],
            "filtered": msm_estimate["filtered"],
            "A": msm_estimate["A"],
            "g_m": msm_estimate["g_m"],
            "optim_message": opt_result.message,
            "optim_convergence": opt_result.success,
            "optim_iter": opt_result.nit,
            "para": self.para,
            "se": self.se,
            "coefficients": coef
        }

    def summary(self):
        se = self.se
        tval = self.results["coefficients"] / se
        p_value = 2 * (1 - stats.t.cdf(np.abs(tval), df=len(self.ret)-4))

        print("*----------------------------------------------------------------------------*")
        print(f"  Markov Switching Multifractal Model With {self.kbar} Volatility Component(s)")
        print("*----------------------------------------------------------------------------*")
        print()

        summary_data = {
            "Estimate": self.results["coefficients"],
            "Std Error": se.flatten(),
            "t-value": tval.flatten(),
            "p-value": p_value.flatten()
        }

        summary_df = pd.DataFrame(summary_data, index=self.coef_names)

        print(summary_df.round(4))
        print(f"\nLogLikelihood: {self.results['LL']}")

        return summary_df

    def predict(self, h=None):
        """
        Predict conditional volatility.

        Parameters:
        -----------
        :param h : int, optional
            Forecast horizon. If None, returns fitted values.

        :return: dict
            Dictionary with volatility and squared volatility predictions
        """
        if h is not None and h < 1:
            raise ValueError("h must be a non-zero integer")

        return msm_predict(
            self.results["g_m"],
            self.para[3],
            self.n_vol,
            self.results["filtered"],
            self.results["A"],
            h
        )

    def plot(self, what="vol"):
        if what not in ["vol", "volsq"]:
            raise ValueError("'what' must be either 'vol' or 'volsq'")

        smoothed_p = msm_smooth(self.results["A"], self.results["filtered"])

        pred = msm_predict(
            self.results["g_m"],
            self.para[3],
            self.n_vol,
            smoothed_p,
            self.results["A"]
        )

        plt.figure(figsize=(12, 6))

        if what == "vol":
            plt.plot(pred["vol"], label="Conditional Volatility")
            plt.plot(np.abs(self.ret), label="Absolute Returns", alpha=0.5)
            plt.title("Conditional Volatility vs Absolute Returns")
            plt.ylabel("Volatility")
        else:  # volsq
            plt.plot(pred["vol_sq"], label="Conditional Variance")
            plt.plot(np.square(self.ret), label="Squared Returns", alpha=0.5)
            plt.title("Conditional Variance vs Squared Returns")
            plt.ylabel("Variance")

        plt.xlabel("Time")
        plt.legend()
        plt.show()

    def decompose(self):
        """
        Decompose returns into volatility components.

        :returns:
        numpy.ndarray
            Matrix of volatility components
        """
        if self.kbar < 2:
            raise ValueError("k (number of volatility components) must be > 1")

        m0 = self.para[0]
        p = msm_smooth(self.results["A"], self.results["filtered"])
        m_mat = msm_clustermat(m0, self.kbar)
        m_marginals = msm_marginals(p, m0, m_mat, self.kbar)

        # Expected values
        em = m0 * m_marginals + (2 - m0) * (1 - m_marginals)

        return em

    def plot_components(self):
        if self.kbar < 2:
            raise ValueError("k (number of volatility components) must be > 1")

        em = self.decompose()

        fig, axes = plt.subplots(self.kbar, 1, figsize=(12, 3*self.kbar), sharex=True)

        for k in range(self.kbar):
            if self.kbar > 1:
                ax = axes[k]
            else:
                ax = axes

            ax.plot(em[:, k])
            ax.set_title(f"Volatility Component M{k+1}")
            ax.set_ylabel("M")

        plt.tight_layout()
        plt.show()