import pandas as pd
import numpy as np


class UnivariateDataloader:
    def __init__(self, df: pd.DataFrame, price_col: str = 'trade_price', mode: str = 'log'):
        df = df.copy()

        df.index.name = 'Date'
        df.index = pd.to_datetime(df.index)

        df = df.sort_index()
        df['log_return'] = np.log(df[price_col]).diff()
        df = df.dropna(subset=['log_return'])

        # Store processed data
        self.data = df

    def get_data(self) -> pd.DataFrame:
        return self.data

