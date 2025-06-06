import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class MSMValidator:
    """
    Evaluate and validate a fitted MSM model.

    Parameters
    ----------
    model : MSM
        A fitted MSM instance.
    data_df : pd.DataFrame
        DataFrame that contains at least the column `return_col`
        (e.g. log-returns) and a DateTime index.
    return_col : str, default 'log_return'
        Name of the column holding the returns used to fit `model`.

    Notes
    -----
    * In-sample = “smoothed” conditional volatility/variance from the model
      compared with realised single-period measures (|r_t| or r_t²).
    * One-step forecast = h = 1 prediction computed from *filtered*
      probabilities at time t–1 compared with the observation at t.
    * All annualisation uses the same `n_vol` factor you passed to the MSM.
    """

    # ------------------------------------------------------------------ #
    # INIT & BASIC PREP
    # ------------------------------------------------------------------ #

    def __init__(self, model, data_df: pd.DataFrame, return_col: str = 'log_return'):

        if return_col not in data_df.columns:
            raise ValueError(f"Column '{return_col}' not found in data_df")

        self.model = model
        self.data = data_df.copy()
        self.return_col = return_col
        self.n_vol = int(model.n_vol)

        # numpy (T,) array of returns
        self.r = self.data[return_col].values.astype(float)

        # smoothed (in-sample) conditional vol & var
        fitted = self.model.predict(h=None)
        self.vol_fit = fitted['vol'].flatten()          # annualised σ̂_t
        self.var_fit = fitted['vol_sq'].flatten()       # annualised σ̂²_t

        # realised single-period measures
        self.var_real = (self.r ** 2) * self.n_vol      # r_t² × n_vol
        self.vol_real = np.abs(self.r) * np.sqrt(self.n_vol)

        # one-step-ahead forecasts (arrays the same length as data, 0 = NaN)
        self.vol_fcst, self.var_fcst = self._make_one_step_forecasts()

    # ------------------------------------------------------------------ #
    # ONE-STEP FORECASTS
    # ------------------------------------------------------------------ #

    def _make_one_step_forecasts(self):
        """
        Produce 1-step-ahead forecasts from filtered probabilities.

        Returns
        -------
        tuple of np.ndarray
            (forecast_vol_ann, forecast_var_ann) each length T; index 0 = NaN
        """
        P_filt = self.model.results['filtered_probabilities']      # (T,k)
        A = self.model.results['transition_matrix']                # (k,k)
        g_m = self.model.results['state_vol_multipliers'].flatten()
        sigma2 = self.model.parameters[3] ** 2                     # unannualised σ²
        g_m_sq = g_m ** 2

        T, _ = P_filt.shape
        var_fcst = np.full(T, np.nan)
        vol_fcst = np.full(T, np.nan)

        for t in range(T - 1):
            p_next = P_filt[t] @ A            # P(S_{t+1}|info_t)
            var_unann = sigma2 * (p_next @ g_m_sq)
            var_ann = var_unann * self.n_vol
            var_fcst[t + 1] = var_ann
            vol_fcst[t + 1] = np.sqrt(var_ann)

        return vol_fcst, var_fcst

    # ------------------------------------------------------------------ #
    # METRICS
    # ------------------------------------------------------------------ #

    @staticmethod
    def _qlike(realised_var, forecast_var, eps=1e-16):
        ratio = realised_var / (forecast_var + eps)
        return np.mean(ratio - np.log(ratio) - 1.0)

    def _error_stats(self, realised, fitted):
        diff = fitted - realised
        mse = np.mean(diff ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(diff))
        return mse, rmse, mae

    def in_sample_metrics(self):
        """
        Returns
        -------
        dict
            MSE, RMSE, MAE for variance and volatility, plus QLIKE on variance.
        """
        mse_v, rmse_v, mae_v = self._error_stats(self.var_real, self.var_fit)
        mse_s, rmse_s, mae_s = self._error_stats(self.vol_real, self.vol_fit)
        return {
            'MSE_var': mse_v,
            'RMSE_var': rmse_v,
            'MAE_var': mae_v,
            'MSE_vol': mse_s,
            'RMSE_vol': rmse_s,
            'MAE_vol': mae_s,
            'QLIKE': self._qlike(self.var_real, self.var_fit)
        }

    def forecast_metrics(self):
        """
        Returns
        -------
        dict
            Same metrics as `in_sample_metrics` but for 1-step forecasts.
        """
        mask = ~np.isnan(self.var_fcst)  # skip first NaN forecast
        mse_v, rmse_v, mae_v = self._error_stats(self.var_real[mask],
                                                 self.var_fcst[mask])
        mse_s, rmse_s, mae_s = self._error_stats(self.vol_real[mask],
                                                 self.vol_fcst[mask])
        return {
            'MSE_var': mse_v,
            'RMSE_var': rmse_v,
            'MAE_var': mae_v,
            'MSE_vol': mse_s,
            'RMSE_vol': rmse_s,
            'MAE_vol': mae_s,
            'QLIKE': self._qlike(self.var_real[mask], self.var_fcst[mask])
        }


    def plot_fitted(self):
        """Smoothed in-sample σ̂_t vs |r_t|√n_vol."""
        plt.figure(figsize=(12, 5))
        plt.plot(self.data.index, self.vol_fit, label='Fitted σ (annualised)')
        plt.plot(self.data.index, self.vol_real, ':',
                 label='|r_t| × √n_vol', alpha=0.6)
        plt.title('In-sample conditional volatility vs realised')
        plt.ylabel('Volatility')
        plt.legend()
        plt.grid(alpha=0.4)
        plt.tight_layout()
        plt.show()

    def plot_forecast(self):
        """One-step-ahead forecast σ̂_{t+1|t} vs realised."""
        plt.figure(figsize=(12, 5))
        plt.plot(self.data.index, self.vol_fcst,
                 label='1-step forecast σ (annualised)')
        plt.plot(self.data.index, self.vol_real, ':',
                 label='|r_t| × √n_vol', alpha=0.6)
        plt.title('One-step-ahead forecast vs realised volatility')
        plt.ylabel('Volatility')
        plt.legend()
        plt.grid(alpha=0.4)
        plt.tight_layout()
        plt.show()