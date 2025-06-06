import pandas as pd
import numpy as np


class UnivariateDataloader:
    def __init__(self, df: pd.DataFrame, price_col: str = 'trade_price'):
        # Copy input to avoid modifying the original
        df = df.copy()

        # Rename index to 'Date' and convert to datetime
        df.index.name = 'Date'
        df.index = pd.to_datetime(df.index)

        # Sort by Date
        df = df.sort_index()

        df['raw_return'] = df[price_col].pct_change()

        df['log_return'] = np.log(df[price_col]).diff()

        df['squared_return'] = df['log_return'] ** 2

        df = df.dropna(subset=['raw_return', 'log_return'])

        # Store the processed DataFrame
        self._data = df

    def get_dataframe(self) -> pd.DataFrame:
        """
        Returns the processed DataFrame with columns:
        - 'trade_price'
        - 'raw_return'
        - 'log_return'
        - 'squared_return'
        """
        return self._data

    def get_raw_returns_array(self) -> np.ndarray:
        """
        Returns the raw returns as a numpy array of shape (T, 1),
        ready for modeling if needed.
        """
        raw = self._data['raw_return'].values.reshape(-1, 1)
        return raw

    def get_log_returns_array(self) -> np.ndarray:
        """
        Returns the log returns as a numpy array of shape (T, 1),
        ready for modeling (e.g., passing to MSM).
        """
        log_r = self._data['log_return'].values.reshape(-1, 1)
        return log_r

    def get_squared_returns_array(self) -> np.ndarray:
        """
        Returns the squared log returns as a numpy array of shape (T, 1).
        """
        sq = self._data['squared_return'].values.reshape(-1, 1)
        return sq
