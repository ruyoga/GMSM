import pandas as pd
import numpy as np
import polars as pl
from pathlib import Path

class MDSVPreprocessor:
    def __init__(self, output_name):
        self.output_name = output_name
        self.years = range(2015, 2025)
        self.root = Path().resolve().parent
        self.input_path = self.root / 'data' / 'cboe'
        self.prev_year_close = None
        self.all_daily_data = []

        processed_data = self.load()
        self.save(processed_data, self.output_name)

    def save(self, df, output_name):
        output_path = self.root / 'data' / 'processed' / output_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    def load(self) -> pd.DataFrame:
        for year in self.years:
            print(f"Processing year {year}")
            try:
                lazy_df = pl.scan_parquet(self.input_path / str(year))
                df = lazy_df.collect().to_pandas()

                print(f'loaded {len(df)} rows for {year}')

                # Get underlying prices at each unique date

                underlying_prices = (
                    df[['quote_datetime', 'active_underlying_price']]
                    .drop_duplicates(subset=['quote_datetime'])
                    .sort_values('quote_datetime')
                    .reset_index(drop=True)
                )

                del df

                self.transform(underlying_prices)

            except Exception as e:
                print(f'error processing {year}', e)

        combined_daily_data = pd.concat(self.all_daily_data, ignore_index=True)

        del self.all_daily_data

        combined_daily_data = combined_daily_data.sort_values('date').reset_index(drop=True)

        # Demean returns
        mean_return = combined_daily_data['daily_log_return_pct'].mean()
        combined_daily_data['demeaned_log_return'] = combined_daily_data['daily_log_return_pct'] - mean_return
        if pd.isna(combined_daily_data.loc[0, 'daily_log_return_pct']):
            combined_daily_data = combined_daily_data.iloc[1:-1].reset_index(drop=True)
        else:
            combined_daily_data = combined_daily_data.iloc[:-1].reset_index(drop=True)

        return combined_daily_data

    def transform(self, df: pd.DataFrame):

        df['log_return'] = (
            np.log(df['active_underlying_price']) -
            np.log(df['active_underlying_price'].shift(1))
        )

        df['squared_log_return'] = df['log_return'] ** 2
        df['date'] = pd.to_datetime(df['quote_datetime']).dt.date

        daily_data = df.groupby('date').agg({
            'squared_log_return': 'sum',
            'quote_datetime': 'count'
        }).reset_index()

        daily_data.columns = ['date', 'sum_squared_returns', 'n_obs']
        daily_data['realized_variance'] = daily_data['sum_squared_returns'] * (100**2)

        daily_close = df.groupby('date').agg({
            'active_underlying_price': 'last'
        }).reset_index()

        daily_close.columns = ['date', 'close_price']

        if self.prev_year_close is not None:
            first_day_return = (
                np.log(daily_close.loc[0, 'close_price']) -
                np.log(self.prev_year_close)
            )
            daily_close.loc[0, 'daily_log_return'] = first_day_return

        self.prev_year_close = daily_close.loc[len(daily_close)-1, 'close_price']

        daily_data = daily_data.merge(
            daily_close[['date', 'daily_log_return_pct']],
            on='date',
            how='left'
        )

        self.all_daily_data.append(daily_data)

        del df, daily_close







