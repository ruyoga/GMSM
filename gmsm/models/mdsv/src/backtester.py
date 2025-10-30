import polars as pl
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import warnings


class OptionsBacktesterPolars:
    def __init__(
            self,
            predictions_df: Union[pl.DataFrame, pl.LazyFrame],
            options_df: Union[pl.DataFrame, pl.LazyFrame],
            initial_capital: float = 100000,
            strategy_type: str = 'straddle',
            threshold: float = 0.02,
            base_position_size: float = 10000,
            signal_multiplier: float = 2.0,
            maturity_filter: Optional[str] = None,
            moneyness_range: Tuple[float, float] = (0.90, 1.10),
            strangle_strikes: Tuple[float, float] = (0.95, 1.05),
            output_dir: Optional[Path] = None
    ):
        """
        Initialize backtester for LONG-ONLY volatility strategy

        Parameters
        ----------
        predictions_df : Polars DataFrame with MDSV predictions
        options_df : Polars DataFrame with options data
        threshold : Signal threshold - trades when predicted_vol > market_iv + threshold
                    Can be negative to trade when predicted < market (underpriced vol)
        """
        self.initial_capital = initial_capital
        self.strategy_type = strategy_type
        self.threshold = threshold
        self.base_position_size = base_position_size
        self.signal_multiplier = signal_multiplier
        self.maturity_filter = maturity_filter
        self.moneyness_range = moneyness_range
        self.strangle_strikes = strangle_strikes
        self.output_dir = output_dir or Path('data/results')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Convert to LazyFrames if not already
        if isinstance(predictions_df, pl.DataFrame):
            self.predictions_lf = predictions_df.lazy()
        else:
            self.predictions_lf = predictions_df

        if isinstance(options_df, pl.DataFrame):
            self.options_lf = options_df.lazy()
        else:
            self.options_lf = options_df

        # Pre-process data
        self._prepare_data()

        # Risk-free rate schedule for 2024
        self.risk_free_rates = {
            datetime(2024, 1, 1): 0.055,
            datetime(2024, 9, 18): 0.050,
            datetime(2024, 11, 7): 0.0475,
            datetime(2024, 12, 18): 0.045
        }

        self.trades = []
        self.equity_curve = []

    def _prepare_data(self):
        print("Pre-filtering options data...")

        # Process predictions
        # predictions
        self.predictions_lf = (
            self.predictions_lf
            .with_columns([
                pl.col('date')
                .cast(pl.Utf8)
                .str.to_datetime(strict=False, time_zone="UTC")
                .dt.replace_time_zone(None)  # make naive
                .alias('date')
            ])
            .sort('date')
        )

        self.options_lf = (
            self.options_lf
            .with_columns([
                pl.col('quote_datetime')
                .cast(pl.Utf8)
                .str.to_datetime(strict=False, time_zone="UTC")
                .dt.replace_time_zone(None)
                .alias('quote_datetime'),
                pl.col('expiration')
                .cast(pl.Utf8)
                .str.to_datetime(strict=False, time_zone="UTC")
                .dt.replace_time_zone(None)
                .alias('expiration'),
            ])
        )

        # Calculate moneyness if not present
        self.options_lf = self.options_lf.with_columns([
            (pl.col('strike') / pl.col('active_underlying_price')).alias('moneyness')
        ])

        # Calculate days to expiry
        self.options_lf = self.options_lf.with_columns([
            ((pl.col('expiration') - pl.col('quote_datetime')).dt.total_days()).alias('days_to_expiry')
        ])

        # Apply moneyness filter
        self.options_lf = self.options_lf.filter(
            (pl.col('moneyness') >= self.moneyness_range[0]) &
            (pl.col('moneyness') <= self.moneyness_range[1])
        )

        # Apply maturity filter
        if self.maturity_filter:
            maturity_days = {'1W': 5, '2W': 10, '1M': 20, '1Q': 60}[self.maturity_filter]
            self.options_lf = self.options_lf.filter(
                (pl.col('days_to_expiry') >= maturity_days - 2) &
                (pl.col('days_to_expiry') <= maturity_days + 2)
            )
        else:
            # Keep reasonable maturities
            self.options_lf = self.options_lf.filter(
                (pl.col('days_to_expiry') >= 3) &
                (pl.col('days_to_expiry') <= 65)
            )

        # Filter low prices
        self.options_lf = self.options_lf.filter(pl.col('mid') >= 0.10)

        # Keep only first quote of each day
        self.options_lf = (
            self.options_lf
            .with_columns([
                pl.col('quote_datetime').dt.date().alias('quote_date')
            ])
            .with_columns([
                pl.col('quote_datetime')
                .min()
                .over('quote_date')
                .alias('first_quote_time')
            ])
            .filter(pl.col('quote_datetime') == pl.col('first_quote_time'))
            .drop(['first_quote_time'])
        )

        # Collect once to report statistics
        options_sample = self.options_lf.select([
            pl.count().alias('total_count'),
            pl.col('quote_date').n_unique().alias('unique_dates')
        ]).collect()

        total_count = options_sample['total_count'][0]
        unique_dates = options_sample['unique_dates'][0]

        print(f"Filtered to {total_count:,} relevant options")
        print(f"  Unique trading dates: {unique_dates}")
        print(f"  Avg options per date: {total_count / unique_dates if unique_dates > 0 else 0:.0f}")

    def _get_risk_free_rate(self, date: datetime) -> float:
        """Get risk-free rate for a given date"""
        rate = 0.055
        for cutoff_date, r in sorted(self.risk_free_rates.items()):
            if date >= cutoff_date:
                rate = r
        return rate

    def _black_scholes_price(self, S: float, K: float, T: float, r: float,
                             sigma: float, option_type: str) -> float:
        if T <= 0:
            if option_type.lower() == 'c':
                return max(S - K, 0)
            else:
                return max(K - S, 0)

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type.lower() == 'c':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        return price

    def _implied_volatility_vectorized(self, prices: np.ndarray, S: float,
                                       strikes: np.ndarray, T: float, r: float,
                                       option_types: np.ndarray) -> np.ndarray:
        ivs = np.full_like(prices, np.nan)

        for i in range(len(prices)):
            if T <= 0:
                continue

            option_type = option_types[i]
            K = strikes[i]
            price = prices[i]

            intrinsic = max(S - K, 0) if option_type.lower() == 'c' else max(K - S, 0)

            if price <= intrinsic:
                continue

            def objective(sigma):
                return self._black_scholes_price(S, K, T, r, sigma, option_type) - price

            try:
                iv = brentq(objective, 0.0001, 5.0, maxiter=100)
                if 0.01 <= iv <= 1.0:
                    ivs[i] = iv
            except (ValueError, RuntimeError):
                continue

        return ivs

    def _get_predictions_for_horizon(self, current_date: datetime,
                                     horizon_days: int) -> Tuple[
        Optional[float], Optional[float], List[float], List[float]]:
        """
        Get cumulative predicted and actual RV for a given horizon

        Returns
        -------
        predicted_cum_rv : float or None
        actual_cum_rv : float or None
        predicted_rvs : list of floats
        actual_rvs : list of floats
        """
        # Collect predictions for this specific date
        predictions_df = self.predictions_lf.collect()

        current_quarter = predictions_df.filter(
            pl.col('date') == current_date
        ).select('quarter')

        if len(current_quarter) == 0:
            return None, None, [], []

        current_quarter = current_quarter['quarter'][0]

        quarter_data = (
            predictions_df
            .filter(pl.col('quarter') == current_quarter)
            .sort('date')
        )

        # Find position in quarter
        all_dates = quarter_data['date'].to_list()
        if current_date not in all_dates:
            return None, None, [], []

        current_idx = all_dates.index(current_date)

        # Get predictions ahead
        predictions_ahead = quarter_data['predicted_rv'][current_idx:current_idx + horizon_days]
        actuals_ahead = quarter_data['actual_rv'][current_idx:current_idx + horizon_days]

        if len(predictions_ahead) < horizon_days:
            return None, None, [], []

        cumulative_predicted_rv = predictions_ahead.sum()
        cumulative_actual_rv = actuals_ahead.sum()

        return (
            float(cumulative_predicted_rv),
            float(cumulative_actual_rv),
            predictions_ahead.to_list(),
            actuals_ahead.to_list()
        )

    def _convert_rv_to_vol(self, cumulative_rv: float, n_days: int) -> float:
        """Convert cumulative RV to annualized volatility"""
        avg_daily_rv = cumulative_rv / n_days
        annualized_vol = np.sqrt(avg_daily_rv * 252) / 100
        return annualized_vol

    def _get_options_for_date(self, trade_date: datetime, days_to_expiry: int) -> pl.DataFrame:
        options = (
            self.options_lf
            .filter(
                (pl.col('quote_date') == trade_date.date()) &
                (pl.col('days_to_expiry') >= days_to_expiry - 2) &
                (pl.col('days_to_expiry') <= days_to_expiry + 2)
            )
            .collect()
        )

        return options

    def _calculate_market_iv(self, options: pl.DataFrame, trade_date: datetime,
                             horizon_days: int) -> Optional[float]:
        """Calculate average market implied volatility"""
        if len(options) == 0:
            return None

        S = options['active_underlying_price'][0]
        T = horizon_days / 365.0
        r = self._get_risk_free_rate(trade_date)

        prices = options['mid'].to_numpy()
        strikes = options['strike'].to_numpy()
        option_types = options['option_type'].to_numpy()

        # Calculate IVs vectorized
        ivs = self._implied_volatility_vectorized(prices, S, strikes, T, r, option_types)

        # Filter valid IVs
        valid_ivs = ivs[~np.isnan(ivs)]

        if len(valid_ivs) == 0:
            return None

        return float(np.mean(valid_ivs))

    def _should_trade_long(self, predicted_vol: float, market_iv: float) -> bool:
        """
        Check if we should enter a LONG volatility trade

        Signal = predicted_vol - market_iv
        Trade when signal > threshold (volatility is underpriced)

        """
        signal_strength = predicted_vol - market_iv
        return signal_strength > self.threshold

    def _calculate_position_multiplier(self, signal_strength: float) -> float:
        """
        Calculate position multiplier using Option B: Directional Signal Excess

        Scales based on how much signal exceeds threshold in the trading direction.
        Works naturally with negative thresholds.

        Parameters
        ----------
        signal_strength : float
            predicted_vol - market_iv

        Returns
        -------
        float
            Position multiplier between 0 and signal_multiplier
        """
        # Calculate excess signal beyond threshold
        signal_excess = signal_strength - self.threshold

        # Scale from 0 at threshold to max at (threshold + |threshold|)
        # This gives 1x at threshold and scales linearly
        if abs(self.threshold) > 1e-6:  # Avoid division by zero
            position_multiplier = signal_excess / abs(self.threshold)
        else:
            # If threshold is ~0, use a default scaling
            position_multiplier = abs(signal_excess) / 0.01

        # Clamp between 0 and signal_multiplier
        position_multiplier = min(max(position_multiplier, 0), self.signal_multiplier)

        return position_multiplier

    def _execute_straddle(self, trade_date: datetime, options: pl.DataFrame,
                          predicted_vol: float, signal_strength: float,
                          horizon_days: int) -> Optional[Dict]:
        """Execute ATM straddle (LONG only)"""
        S = options['active_underlying_price'][0]

        # Split into calls and puts
        calls = options.filter(pl.col('option_type').str.to_lowercase() == 'c')
        puts = options.filter(pl.col('option_type').str.to_lowercase() == 'p')

        if len(calls) == 0 or len(puts) == 0:
            return None

        # Find ATM strike
        call_strikes = calls['strike'].to_numpy()
        atm_strike = call_strikes[np.argmin(np.abs(call_strikes - S))]

        # Get specific options
        call_option = calls.filter(pl.col('strike') == atm_strike)
        put_option = puts.filter(pl.col('strike') == atm_strike)

        if len(call_option) == 0 or len(put_option) == 0:
            return None

        call_option = call_option[0]
        put_option = put_option[0]

        # Calculate IVs for verification
        T = call_option['days_to_expiry'][0] / 365.0
        r = self._get_risk_free_rate(trade_date)

        call_iv = self._implied_volatility_vectorized(
            np.array([call_option['mid'][0]]), S,
            np.array([atm_strike]), T, r, np.array(['c'])
        )[0]

        put_iv = self._implied_volatility_vectorized(
            np.array([put_option['mid'][0]]), S,
            np.array([atm_strike]), T, r, np.array(['p'])
        )[0]

        if np.isnan(call_iv) or np.isnan(put_iv):
            return None

        avg_iv = (call_iv + put_iv) / 2.0

        # Position sizing with new scaling method
        position_multiplier = self._calculate_position_multiplier(signal_strength)
        position_size = self.base_position_size * position_multiplier

        call_premium = call_option['mid'][0]
        put_premium = put_option['mid'][0]
        total_premium = call_premium + put_premium

        # SPX multiplier is 100
        n_contracts = position_size / (total_premium * 100)

        if n_contracts < 0.01:
            return None

        return {
            'entry_date': trade_date,
            'strategy': 'straddle',
            'direction': 'long',  # Always long
            'expiration': call_option['expiration'][0],
            'days_to_expiry': horizon_days,
            'underlying_price_entry': S,
            'strike': atm_strike,
            'call_strike': atm_strike,
            'put_strike': atm_strike,
            'call_premium': call_premium,
            'put_premium': put_premium,
            'total_premium': total_premium,
            'n_contracts': n_contracts,
            'position_size': position_size,
            'position_multiplier': position_multiplier,
            'predicted_vol': predicted_vol,
            'implied_vol': avg_iv,
            'signal_strength': signal_strength,
            'risk_free_rate': r
        }

    def _execute_strangle(self, trade_date: datetime, options: pl.DataFrame,
                          predicted_vol: float, signal_strength: float,
                          horizon_days: int) -> Optional[Dict]:
        """Execute OTM strangle (LONG only)"""
        S = options['active_underlying_price'][0]

        # Split into calls and puts
        calls = options.filter(pl.col('option_type').str.to_lowercase() == 'c')
        puts = options.filter(pl.col('option_type').str.to_lowercase() == 'p')

        if len(calls) == 0 or len(puts) == 0:
            return None

        # Find OTM strikes
        put_target_strike = S * self.strangle_strikes[0]
        call_target_strike = S * self.strangle_strikes[1]

        put_strikes = puts['strike'].to_numpy()
        call_strikes = calls['strike'].to_numpy()

        put_strike = put_strikes[np.argmin(np.abs(put_strikes - put_target_strike))]
        call_strike = call_strikes[np.argmin(np.abs(call_strikes - call_target_strike))]

        # Get specific options
        put_option = puts.filter(pl.col('strike') == put_strike)
        call_option = calls.filter(pl.col('strike') == call_strike)

        if len(call_option) == 0 or len(put_option) == 0:
            return None

        call_option = call_option[0]
        put_option = put_option[0]

        call_premium = call_option['mid'][0]
        put_premium = put_option['mid'][0]

        if put_premium < 0.04 or call_premium < 0.04:
            return None

        # Calculate IVs
        T = horizon_days / 365.0
        r = self._get_risk_free_rate(trade_date)

        call_iv = self._implied_volatility_vectorized(
            np.array([call_premium]), S,
            np.array([call_strike]), T, r, np.array(['c'])
        )[0]

        put_iv = self._implied_volatility_vectorized(
            np.array([put_premium]), S,
            np.array([put_strike]), T, r, np.array(['p'])
        )[0]

        if np.isnan(call_iv) or np.isnan(put_iv):
            return None

        avg_iv = (call_iv + put_iv) / 2.0

        # Position sizing with new scaling method
        position_multiplier = self._calculate_position_multiplier(signal_strength)
        position_size = self.base_position_size * position_multiplier

        total_premium = call_premium + put_premium
        n_contracts = position_size / (total_premium * 100)

        if n_contracts < 0.01:
            return None

        return {
            'entry_date': trade_date,
            'strategy': 'strangle',
            'direction': 'long',  # Always long
            'expiration': call_option['expiration'][0],
            'days_to_expiry': horizon_days,
            'underlying_price_entry': S,
            'strike': None,
            'call_strike': call_strike,
            'put_strike': put_strike,
            'call_premium': call_premium,
            'put_premium': put_premium,
            'total_premium': total_premium,
            'n_contracts': n_contracts,
            'position_size': position_size,
            'position_multiplier': position_multiplier,
            'predicted_vol': predicted_vol,
            'implied_vol': avg_iv,
            'signal_strength': signal_strength,
            'risk_free_rate': r
        }

    def _calculate_payoff(self, trade: Dict, expiration_price: float) -> Dict:
        """
        Calculate P&L for LONG options positions at expiration

        Long position: Pay premium upfront, receive intrinsic value at expiration
        """
        call_strike = trade['call_strike']
        put_strike = trade['put_strike']
        n_contracts = trade['n_contracts']

        # Intrinsic values at expiration
        call_intrinsic = max(expiration_price - call_strike, 0)
        put_intrinsic = max(put_strike - expiration_price, 0)

        # Total values with multiplier
        call_value = call_intrinsic * 100 * n_contracts
        put_value = put_intrinsic * 100 * n_contracts
        total_intrinsic_value = call_value + put_value

        # Premium paid
        total_premium_value = trade['total_premium'] * 100 * n_contracts

        # P&L: receive intrinsic, paid premium
        net_pnl = total_intrinsic_value - total_premium_value
        return_pct = (net_pnl / total_premium_value) * 100 if total_premium_value > 0 else 0

        trade_result = trade.copy()
        trade_result.update({
            'exit_date': trade['expiration'],
            'underlying_price_exit': expiration_price,
            'call_intrinsic': call_intrinsic,
            'put_intrinsic': put_intrinsic,
            'total_intrinsic_value': total_intrinsic_value,
            'total_premium_value': total_premium_value,
            'net_pnl': net_pnl,
            'return_pct': return_pct
        })

        return trade_result

    def _get_expiration_price(self, expiration_date: datetime) -> Optional[float]:
        """Get underlying price at expiration"""
        expiration_prices = (
            self.options_lf
            .filter(pl.col('quote_date') == expiration_date.date())
            .select([
                pl.col('quote_datetime').max().alias('last_quote_time'),
                pl.col('active_underlying_price').first().alias('price')
            ])
            .collect()
        )

        if len(expiration_prices) == 0:
            return None

        # Get the last price of the day
        last_prices = (
            self.options_lf
            .filter(
                (pl.col('quote_date') == expiration_date.date()) &
                (pl.col('quote_datetime') == expiration_prices['last_quote_time'][0])
            )
            .select('active_underlying_price')
            .collect()
        )

        if len(last_prices) == 0:
            return None

        return float(last_prices['active_underlying_price'][0])

    def run_backtest(self) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        print("\n" + "=" * 60)
        print("STARTING BACKTEST - LONG ONLY VOLATILITY STRATEGY")
        print("=" * 60)
        print(f"Strategy: {self.strategy_type}")
        print(f"Threshold: {self.threshold:.2%}")
        print(f"Base position: ${self.base_position_size:,.0f}")
        print(f"Signal multiplier: {self.signal_multiplier}x")
        print(f"Initial capital: ${self.initial_capital:,.0f}")
        print("-" * 60)

        current_capital = self.initial_capital

        # Collect predictions for iteration
        predictions_df = self.predictions_lf.collect()

        self.equity_curve = [{
            'date': predictions_df['date'].min(),
            'equity': current_capital
        }]

        # Track IV data for analysis
        self.iv_data = []

        maturity_days_map = {'1W': 5, '2W': 10, '1M': 20, '1Q': 60}

        if self.maturity_filter:
            maturities_to_trade = {
                self.maturity_filter: maturity_days_map[self.maturity_filter]
            }
        else:
            maturities_to_trade = maturity_days_map

        total_dates = len(predictions_df)
        dates_with_options = 0
        dates_with_ivs = 0
        dates_with_signal = 0

        # Optimized iteration using iter_rows
        for idx, row in enumerate(predictions_df.iter_rows(named=True)):
            trade_date = row['date']

            if idx % 50 == 0:
                print(f"Progress: {idx}/{total_dates} ({idx / total_dates * 100:.1f}%)")

            for maturity_name, horizon_days in maturities_to_trade.items():
                # Get both predicted and actual RV
                cum_predicted_rv, cum_actual_rv, pred_list, actual_list = self._get_predictions_for_horizon(
                    trade_date, horizon_days
                )

                if cum_predicted_rv is None or cum_actual_rv is None:
                    continue

                # Convert both to annualized volatilities
                predicted_vol = self._convert_rv_to_vol(cum_predicted_rv, horizon_days)
                actual_vol = self._convert_rv_to_vol(cum_actual_rv, horizon_days)

                options = self._get_options_for_date(trade_date, horizon_days)

                if len(options) == 0:
                    continue

                dates_with_options += 1

                # Calculate market IV
                market_iv = self._calculate_market_iv(options, trade_date, horizon_days)

                if market_iv is None:
                    continue

                dates_with_ivs += 1

                # Calculate signal
                signal_strength = predicted_vol - market_iv

                # Track IV data for analysis (record all days with valid IV)
                self.iv_data.append({
                    'date': trade_date,
                    'maturity': maturity_name,
                    'horizon_days': horizon_days,
                    'predicted_vol': predicted_vol,
                    'actual_vol': actual_vol,  # NEW: actual realized volatility
                    'market_iv': market_iv,
                    'signal_strength': signal_strength,
                    'threshold': self.threshold,
                    'traded': False  # Will update if we actually trade
                })

                # Debug output for first few dates
                if idx < 3:
                    print(f"\n  Date {idx + 1}: {trade_date.date()}")
                    print(f"    Options found: {len(options)}")
                    print(f"    Predicted vol: {predicted_vol * 100:.2f}%")
                    print(f"    Actual vol: {actual_vol * 100:.2f}%")
                    print(f"    Market IV: {market_iv * 100:.2f}%")
                    print(f"    Signal: {signal_strength * 100:.2f}%")
                    print(f"    Threshold: {self.threshold * 100:.2f}%")

                # Check if we should trade LONG
                should_trade = self._should_trade_long(predicted_vol, market_iv)

                if not should_trade:
                    continue

                dates_with_signal += 1

                # Mark that we traded on this day
                self.iv_data[-1]['traded'] = True

                # Calculate position multiplier for debugging
                pos_mult = self._calculate_position_multiplier(signal_strength)

                if idx < 3:
                    print(f"    ✓ Signal triggers LONG trade! (Position mult: {pos_mult:.2f}x)")

                # Execute trades
                if self.strategy_type in ['straddle', 'both']:
                    trade = self._execute_straddle(
                        trade_date, options, predicted_vol,
                        signal_strength, horizon_days
                    )
                    if trade:
                        self.trades.append(trade)
                        if idx < 3:
                            print(
                                f"    ✓ Straddle (LONG): {trade['n_contracts']:.2f} contracts @ ${trade['total_premium']:.2f}")

                if self.strategy_type in ['strangle', 'both']:
                    trade = self._execute_strangle(
                        trade_date, options, predicted_vol,
                        signal_strength, horizon_days
                    )
                    if trade:
                        self.trades.append(trade)
                        if idx < 3:
                            print(
                                f"    ✓ Strangle (LONG): {trade['n_contracts']:.2f} contracts @ ${trade['total_premium']:.2f}")

        print(f"\n{'=' * 60}")
        print("BACKTEST DIAGNOSTICS")
        print(f"{'=' * 60}")
        print(f"Dates checked: {total_dates}")
        print(f"Dates with options: {dates_with_options}")
        print(f"Dates with valid IVs: {dates_with_ivs}")
        print(f"Dates with signal triggering: {dates_with_signal}")
        print(f"Total trades entered: {len(self.trades)}")
        print(f"{'=' * 60}\n")

        # Calculate payoffs
        completed_trades = []
        for trade in self.trades:
            expiration_price = self._get_expiration_price(trade['expiration'])

            if expiration_price is None:
                continue

            trade_result = self._calculate_payoff(trade, expiration_price)
            completed_trades.append(trade_result)

            current_capital += trade_result['net_pnl']
            self.equity_curve.append({
                'date': trade_result['exit_date'],
                'equity': current_capital,
                'trade_pnl': trade_result['net_pnl']
            })

        # Convert to Polars DataFrames
        if len(completed_trades) > 0:
            trades_df = pl.DataFrame(completed_trades)
        else:
            trades_df = pl.DataFrame()

        equity_df = pl.DataFrame(self.equity_curve).sort('date')

        # Convert IV data to DataFrame
        if len(self.iv_data) > 0:
            iv_data_df = pl.DataFrame(self.iv_data)
        else:
            iv_data_df = pl.DataFrame()

        # Save results
        if len(trades_df) > 0:
            suffix = self.maturity_filter or "all"
            trades_path = self.output_dir / f'trades_{self.strategy_type}_{suffix}.csv'
            equity_path = self.output_dir / f'equity_{self.strategy_type}_{suffix}.csv'
            iv_path = self.output_dir / f'iv_data_{self.strategy_type}_{suffix}.csv'

            trades_df.write_csv(str(trades_path))
            equity_df.write_csv(str(equity_path))

            if len(iv_data_df) > 0:
                iv_data_df.write_csv(str(iv_path))
                print(f"\n✓ Saved trades to: {trades_path}")
                print(f"✓ Saved equity curve to: {equity_path}")
                print(f"✓ Saved IV data to: {iv_path}")
            else:
                print(f"\n✓ Saved trades to: {trades_path}")
                print(f"✓ Saved equity curve to: {equity_path}")

        self._print_summary(trades_df, current_capital)

        return trades_df, equity_df, iv_data_df

    def _print_summary(self, trades_df: pl.DataFrame, final_capital: float):
        """Print backtest summary statistics"""
        if len(trades_df) == 0:
            print("\nNo trades executed!")
            return

        # Calculate statistics
        net_pnls = trades_df['net_pnl'].to_numpy()
        return_pcts = trades_df['return_pct'].to_numpy()

        winners = np.sum(net_pnls > 0)
        losers = np.sum(net_pnls < 0)

        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Capital: ${final_capital:,.2f}")
        print(f"Total P&L: ${final_capital - self.initial_capital:,.2f}")
        print(f"Return: {((final_capital / self.initial_capital) - 1) * 100:.2f}%")

        print(f"\nTrades: {len(trades_df)}")
        print(f"Winners: {winners} ({winners / len(trades_df) * 100:.1f}%)")
        print(f"Losers: {losers} ({losers / len(trades_df) * 100:.1f}%)")

        print(f"\nAvg P&L per trade: ${np.mean(net_pnls):,.2f}")
        print(f"Avg return per trade: {np.mean(return_pcts):.2f}%")
        print(f"Best trade: ${np.max(net_pnls):,.2f}")
        print(f"Worst trade: ${np.min(net_pnls):,.2f}")

        if np.std(return_pcts) > 0:
            sharpe = np.mean(return_pcts) / np.std(return_pcts)
            print(f"\nSharpe ratio: {sharpe:.2f}")

        print("=" * 60)