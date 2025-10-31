import polars as pl
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field
import time


@dataclass
class Position:
    """Track open position with all necessary data"""
    position_id: str
    entry_date: datetime
    entry_time: datetime
    strategy: str

    # Option details
    call_strike: float
    put_strike: float
    expiration: datetime
    days_to_expiry: int
    n_contracts: float

    # Entry Greeks and pricing
    entry_iv: float
    entry_vega: float
    entry_theta: float
    entry_premium: float
    entry_underlying: float
    predicted_vol: float
    entry_signal: float

    # Tracking
    status: str = 'open'
    tracking_history: List[Dict] = field(default_factory=list)

    # Peak tracking
    peak_iv: float = None
    peak_iv_time: datetime = None
    peak_pnl: float = None
    peak_pnl_time: datetime = None

    # Exit info
    exit_date: datetime = None
    exit_time: datetime = None
    exit_reason: str = None
    exit_iv: float = None
    exit_vega: float = None
    exit_theta: float = None
    exit_pnl: float = None
    exit_underlying: float = None

    def update_peaks(self, current_time: datetime, current_iv: float, current_pnl: float):
        """Update peak tracking"""
        if self.peak_iv is None or current_iv > self.peak_iv:
            self.peak_iv = current_iv
            self.peak_iv_time = current_time

        if self.peak_pnl is None or current_pnl > self.peak_pnl:
            self.peak_pnl = current_pnl
            self.peak_pnl_time = current_time


class OptionsBacktesterV2:
    def __init__(
            self,
            predictions_df: Union[pl.DataFrame, pl.LazyFrame],
            options_df: Union[pl.DataFrame, pl.LazyFrame],
            initial_capital: float = 100000,
            strategy_type: str = 'straddle',
            entry_threshold: float = 0.01,
            exit_threshold: float = 0.005,
            base_position_size: float = 10000,
            signal_multiplier: float = 2.0,
            maturity_filter: Optional[str] = None,
            moneyness_range: float = 0.2,
            strangle_range: float = 0.2,
            tracking_freq: str = '1h',
            stop_loss_pct: float = 0.30,
            greeks_config: Optional[Dict] = None,
            output_dir: Optional[Path] = None
    ):
        """
        Initialize the enhanced backtester with signal reversal and Greeks exits.

        OPTIMIZED FOR COMPUTE SPEED - Memory is not a constraint.

        Parameters
        ----------
        predictions_df : Polars DataFrame/LazyFrame
            MDSV predictions with columns: date, quarter, predicted_rv, actual_rv
        options_df : Polars DataFrame/LazyFrame
            Options data with columns: quote_datetime, expiration, strike, option_type,
            bid, ask, mid, active_underlying_price
        initial_capital : float, default=100000
            Starting capital for backtest
        strategy_type : str, default='straddle'
            Strategy to trade: 'straddle', 'strangle', or 'both'
        entry_threshold : float, default=0.01
            Entry signal threshold: Trade when predicted_vol > market_iv + entry_threshold
        exit_threshold : float, default=0.005
            Exit signal threshold: Exit when predicted_vol < market_iv - exit_threshold
        base_position_size : float, default=10000
            Base position size in dollars
        signal_multiplier : float, default=2.0
            Maximum position multiplier based on signal strength
        maturity_filter : str or None, default=None
            Filter for specific maturity: '1W', '2W', '1M', '1Q', or None for all
        moneyness_range : float, default=0.2
            Range of moneyness to consider (strike/spot as percentage)
        strangle_range : float, default=0.2
            Range for strangle strikes
        tracking_freq : str, default='1h'
            Position tracking frequency: '30min', '1h', '2h', '4h', 'eod'
        stop_loss_pct : float, default=0.30
            Stop loss percentage (0.30 = -30%)
        greeks_config : dict or None, default=None
            Greeks exit configuration. Defaults to:
            {
                'vega_theta_ratio_min': 1.5,
                'vega_decline_pct': 0.30,
                'force_exit_dte': 2
            }
        output_dir : Path or None, default=None
            Directory to save results. Defaults to 'data/results'
        """
        self.initial_capital = initial_capital
        self.strategy_type = strategy_type
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.base_position_size = base_position_size
        self.signal_multiplier = signal_multiplier
        self.maturity_filter = maturity_filter
        self.moneyness_range = (1 - moneyness_range, 1 + moneyness_range)
        self.strangle_strikes = (1 - strangle_range, 1 + strangle_range)
        self.tracking_freq = tracking_freq
        self.stop_loss_pct = stop_loss_pct
        self.output_dir = output_dir or Path('data/results')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Greeks configuration
        self.greeks_config = greeks_config or {
            'vega_theta_ratio_min': 1.5,
            'vega_decline_pct': 0.30,
            'force_exit_dte': 2
        }

        print(f"\nInitializing Backtester V2 (Optimized for Speed)")
        print(f"Strategy: {self.strategy_type}")
        print(f"Entry threshold: {self.entry_threshold:.3%}")
        print(f"Exit threshold: {self.exit_threshold:.3%}")
        print(f"Stop loss: {self.stop_loss_pct:.1%}")
        print(f"Tracking frequency: {self.tracking_freq}")
        print(f"Greeks config: {self.greeks_config}")

        # Convert to DataFrames immediately and work in memory
        print("\nLoading data into memory...")
        if isinstance(predictions_df, pl.LazyFrame):
            self.predictions_df = predictions_df.collect()
        else:
            self.predictions_df = predictions_df.clone()

        if isinstance(options_df, pl.LazyFrame):
            self.options_df = options_df.collect()
        else:
            self.options_df = options_df.clone()

        # Risk-free rate schedule for 2024
        self.risk_free_rates = {
            datetime(2024, 1, 1): 0.055,
            datetime(2024, 9, 18): 0.050,
            datetime(2024, 11, 7): 0.0475,
            datetime(2024, 12, 18): 0.045
        }

        # Initialize storage
        self.open_positions = {}
        self.closed_positions = []
        self.equity_curve = []
        self.tracking_log = []

        # Precomputed lookup dictionaries
        self.predictions_lookup = {}
        self.options_by_datetime = {}  # NEW: Track by datetime, not just date
        self.expiration_prices = {}

        # Pre-process and precompute everything
        self._prepare_data()

    def _prepare_data(self):
        """
        Pre-process data and precompute all necessary lookups.
        OPTIMIZED: Store options at all timestamps for fast position tracking.
        """
        print("\n" + "=" * 60)
        print("DATA PREPARATION & PRECOMPUTATION (SPEED OPTIMIZED)")
        print("=" * 60)

        # Parse dates
        print("1. Parsing dates and filtering data...")
        self.predictions_df = (
            self.predictions_df
            .with_columns([
                pl.col('date')
                .cast(pl.Utf8)
                .str.to_datetime(strict=False, time_zone="UTC")
                .dt.replace_time_zone(None)
                .alias('date')
            ])
            .sort('date')
        )

        self.options_df = (
            self.options_df
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
            .with_columns([
                (pl.col('strike') / pl.col('active_underlying_price')).alias('moneyness'),
                ((pl.col('expiration') - pl.col('quote_datetime')).dt.total_days()).alias('days_to_expiry'),
                pl.col('quote_datetime').dt.date().alias('quote_date'),
                pl.col('quote_datetime').dt.hour().alias('hour'),
                pl.col('quote_datetime').dt.minute().alias('minute')
            ])
        )

        # Apply filters
        self.options_df = self.options_df.filter(
            (pl.col('moneyness') >= self.moneyness_range[0]) &
            (pl.col('moneyness') <= self.moneyness_range[1]) &
            (pl.col('mid') >= 0.10)
        )

        # Apply maturity filter
        if self.maturity_filter:
            maturity_days = {'1W': 5, '2W': 10, '1M': 20, '1Q': 60}[self.maturity_filter]
            self.options_df = self.options_df.filter(
                (pl.col('days_to_expiry') >= maturity_days - 2) &
                (pl.col('days_to_expiry') <= maturity_days + 2)
            )
        else:
            self.options_df = self.options_df.filter(
                (pl.col('days_to_expiry') >= 3) &
                (pl.col('days_to_expiry') <= 65)
            )

        print(f"   Filtered to {len(self.options_df):,} relevant options")
        print(f"   Unique trading dates: {self.options_df['quote_date'].n_unique()}")
        print(f"   Date range: {self.options_df['quote_date'].min()} to {self.options_df['quote_date'].max()}")

        # OPTIMIZATION: Precompute all predictions
        print("\n2. Precomputing predictions for all horizons...")
        self._precompute_predictions()

        # OPTIMIZATION: Index options by DATETIME and maturity for fast tracking
        print(f"\n3. Indexing options by datetime (tracking_freq={self.tracking_freq})...")
        self._create_options_lookup()

        # OPTIMIZATION: Cache expiration prices
        print("\n4. Caching expiration prices...")
        self._precompute_expiration_prices()

        print("\n" + "=" * 60)
        print("PRECOMPUTATION COMPLETE - Ready for backtesting!")
        print("=" * 60 + "\n")

    def _precompute_predictions(self):
        """Precompute cumulative predictions for all dates and horizons."""
        horizons = [5, 10, 20, 60]  # 1W, 2W, 1M, 1Q
        predictions_with_idx = self.predictions_df.with_row_count('idx')

        for horizon in horizons:
            for quarter in predictions_with_idx['quarter'].unique().sort():
                quarter_df = predictions_with_idx.filter(
                    pl.col('quarter') == quarter
                ).sort('date')

                n_rows = len(quarter_df)

                for i in range(n_rows - horizon + 1):
                    window = quarter_df[i:i + horizon]

                    if len(window) == horizon:
                        date = window['date'][0]
                        cum_pred_rv = window['predicted_rv'].sum()
                        cum_actual_rv = window['actual_rv'].sum()

                        pred_vol = self._convert_rv_to_vol(cum_pred_rv, horizon)
                        actual_vol = self._convert_rv_to_vol(cum_actual_rv, horizon)

                        key = (date, horizon)
                        self.predictions_lookup[key] = {
                            'predicted_vol': pred_vol,
                            'actual_vol': actual_vol,
                            'cum_pred_rv': cum_pred_rv,
                            'cum_actual_rv': cum_actual_rv
                        }

        print(f"   ✓ Pre-computed {len(self.predictions_lookup)} date-horizon pairs")

    def _create_options_lookup(self):
        """
        OPTIMIZED: Create fast lookup by datetime with sampling at tracking_freq.
        Store ALL options data in memory for O(1) access during tracking.
        """
        # Sample options based on tracking frequency
        sampled_options = self._sample_options_by_frequency(self.options_df)

        print(f"   Sampled options data: {len(sampled_options):,} rows")
        print(f"   Original options data: {len(self.options_df):,} rows")
        print(f"   Reduction: {(1 - len(sampled_options) / len(self.options_df)) * 100:.1f}%")

        # Create lookup: (datetime, expiration, strike, option_type) -> option data
        # This allows O(1) lookup for specific options during position tracking
        for row in sampled_options.iter_rows(named=True):
            dt = row['quote_datetime']
            exp = row['expiration']
            strike = row['strike']
            opt_type = row['option_type']

            key = (dt, exp, strike, opt_type)
            self.options_by_datetime[key] = {
                'mid': row['mid'],
                'bid': row['bid'],
                'ask': row['ask'],
                'underlying': row['active_underlying_price'],
                'days_to_expiry': row['days_to_expiry']
            }

        # Also keep first-of-day for entry scanning (separate lookup)
        self.entry_scan_df = (
            self.options_df
            .with_columns([
                pl.col('quote_datetime').min().over('quote_date').alias('first_quote_time')
            ])
            .filter(pl.col('quote_datetime') == pl.col('first_quote_time'))
            .drop('first_quote_time')
        )

        print(f"   ✓ Created {len(self.options_by_datetime)} datetime-option lookups")
        print(f"   ✓ Created entry scan data: {len(self.entry_scan_df)} rows")

    def _sample_options_by_frequency(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Sample options data based on tracking frequency.
        Memory is NOT a constraint, so we keep all necessary timestamps.
        """
        if self.tracking_freq == '30min':
            return df  # Keep all

        # Define sampling logic based on frequency
        freq_map = {
            '1h': (0, 30),  # Keep 9:30, 10:30, 11:30, 12:30, 1:30, 2:30, 3:30
            '2h': (30, 30),  # Keep 9:30, 11:30, 1:30, 3:30
            '4h': (30, 0),  # Keep 9:30, 1:30
            'eod': (0, 0)  # Keep only last quote of day
        }

        if self.tracking_freq == 'eod':
            return df.with_columns([
                pl.col('quote_datetime').max().over('quote_date').alias('last_quote_time')
            ]).filter(pl.col('quote_datetime') == pl.col('last_quote_time')).drop('last_quote_time')

        # For hourly sampling, filter by minute
        if self.tracking_freq == '1h':
            return df.filter(pl.col('minute') == 30)  # Keep HH:30

        if self.tracking_freq == '2h':
            return df.filter(
                ((pl.col('hour') % 2 == 1) & (pl.col('minute') == 30)) |  # 9:30, 11:30, 1:30, 3:30
                ((pl.col('hour') == 9) & (pl.col('minute') == 30))
            )

        if self.tracking_freq == '4h':
            return df.filter(
                ((pl.col('hour') == 9) & (pl.col('minute') == 30)) |  # 9:30
                ((pl.col('hour') == 13) & (pl.col('minute') == 30))  # 1:30
            )

        return df

    def _precompute_expiration_prices(self):
        """Cache expiration prices for payoff calculation."""
        unique_dates = self.options_df['quote_date'].unique().sort()

        for exp_date in unique_dates:
            date_options = self.options_df.filter(pl.col('quote_date') == exp_date)

            if len(date_options) == 0:
                continue

            last_quote_time = date_options['quote_datetime'].max()
            last_price = date_options.filter(
                pl.col('quote_datetime') == last_quote_time
            )['active_underlying_price'].first()

            if last_price is not None:
                self.expiration_prices[exp_date] = float(last_price)

        print(f"   ✓ Cached {len(self.expiration_prices)} expiration prices")

    def _convert_rv_to_vol(self, cumulative_rv: float, n_days: int) -> float:
        """Convert cumulative realized variance to annualized volatility"""
        avg_daily_rv = cumulative_rv / n_days
        annualized_vol = np.sqrt(avg_daily_rv * 252) / 100
        return annualized_vol

    def _get_risk_free_rate(self, date: datetime) -> float:
        """Get risk-free rate for a given date"""
        rate = 0.055
        for cutoff_date, r in sorted(self.risk_free_rates.items()):
            if date >= cutoff_date:
                rate = r
        return rate

    def _black_scholes_price(self, S: float, K: float, T: float, r: float,
                             sigma: float, option_type: str) -> float:
        """Calculate Black-Scholes option price"""
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

    def _black_scholes_greeks(self, S: float, K: float, T: float, r: float,
                              sigma: float, option_type: str) -> Dict[str, float]:
        """
        Calculate Greeks for an option.
        OPTIMIZED: Compute all Greeks in single pass.
        """
        if T <= 0:
            return {'vega': 0.0, 'theta': 0.0, 'delta': 0.0, 'gamma': 0.0}

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        # Vega (same for call and put)
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # Divide by 100 for 1% move

        # Gamma (same for call and put)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

        if option_type.lower() == 'c':
            # Call delta and theta
            delta = norm.cdf(d1)
            theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                     - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        else:
            # Put delta and theta
            delta = -norm.cdf(-d1)
            theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                     + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365

        return {
            'vega': vega,
            'theta': theta,
            'delta': delta,
            'gamma': gamma
        }

    def _implied_volatility_vectorized(self, prices: np.ndarray, S: float,
                                       strikes: np.ndarray, T: float, r: float,
                                       option_types: np.ndarray) -> np.ndarray:
        """Calculate implied volatilities for multiple options"""
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

    def _get_option_price(self, timestamp: datetime, expiration: datetime,
                          strike: float, option_type: str) -> Optional[Dict]:
        """
        Get option price at specific timestamp.
        OPTIMIZED: O(1) lookup from precomputed dictionary.
        """
        key = (timestamp, expiration, strike, option_type)
        return self.options_by_datetime.get(key, None)

    def _calculate_position_greeks(self, position: Position, current_time: datetime,
                                   current_prices: Dict) -> Optional[Dict]:
        """
        Calculate average Greeks for straddle/strangle position.
        Returns: {'vega': float, 'theta': float, 'delta': float, 'gamma': float}
        """
        S = current_prices['underlying']
        T = current_prices['days_to_expiry'] / 365.0
        r = self._get_risk_free_rate(current_time)

        # Get option prices
        call_data = self._get_option_price(
            current_time, position.expiration,
            position.call_strike, 'C'
        )
        put_data = self._get_option_price(
            current_time, position.expiration,
            position.put_strike, 'P'
        )

        if call_data is None or put_data is None:
            return None

        # Calculate IVs
        call_iv = self._implied_volatility_vectorized(
            np.array([call_data['mid']]), S,
            np.array([position.call_strike]), T, r, np.array(['c'])
        )[0]

        put_iv = self._implied_volatility_vectorized(
            np.array([put_data['mid']]), S,
            np.array([position.put_strike]), T, r, np.array(['p'])
        )[0]

        if np.isnan(call_iv) or np.isnan(put_iv):
            return None

        # Calculate Greeks for each leg
        call_greeks = self._black_scholes_greeks(
            S, position.call_strike, T, r, call_iv, 'c'
        )
        put_greeks = self._black_scholes_greeks(
            S, position.put_strike, T, r, put_iv, 'p'
        )

        # Average Greeks (weighted equally for straddle/strangle)
        avg_greeks = {
            'vega': (call_greeks['vega'] + put_greeks['vega']) / 2,
            'theta': (call_greeks['theta'] + put_greeks['theta']) / 2,
            'delta': (call_greeks['delta'] + put_greeks['delta']) / 2,
            'gamma': (call_greeks['gamma'] + put_greeks['gamma']) / 2,
            'call_iv': call_iv,
            'put_iv': put_iv,
            'avg_iv': (call_iv + put_iv) / 2
        }

        return avg_greeks

    def _calculate_position_pnl(self, position: Position, current_prices: Dict) -> float:
        """
        Calculate current P&L for position.
        """
        # Get current option prices
        call_data = self._get_option_price(
            current_prices['timestamp'], position.expiration,
            position.call_strike, 'C'
        )
        put_data = self._get_option_price(
            current_prices['timestamp'], position.expiration,
            position.put_strike, 'P'
        )

        if call_data is None or put_data is None:
            return 0.0

        current_value = (call_data['mid'] + put_data['mid']) * 100 * position.n_contracts
        entry_value = position.entry_premium * 100 * position.n_contracts

        return current_value - entry_value

    def _check_exit_conditions(self, position: Position, current_time: datetime,
                               current_greeks: Dict, current_pnl: float) -> Tuple[bool, str]:
        """
        Check if position should be exited.

        Priority order:
        1. Risk management (stop loss, expiration)
        2. Signal reversal
        3. Greeks deterioration

        Returns: (should_exit, exit_reason)
        """
        days_to_expiry = current_greeks.get('days_to_expiry', 0)

        # PRIORITY 1: Risk management
        entry_value = position.entry_premium * 100 * position.n_contracts
        pnl_pct = current_pnl / entry_value if entry_value > 0 else 0

        if pnl_pct < -self.stop_loss_pct:
            return True, f'stop_loss_{pnl_pct:.2%}'

        if days_to_expiry <= self.greeks_config['force_exit_dte']:
            return True, f'expiration_dte_{days_to_expiry}'

        # PRIORITY 2: Signal reversal
        current_signal = position.predicted_vol - current_greeks['avg_iv']
        if current_signal < -self.exit_threshold:
            return True, f'signal_reversal_{current_signal:.4f}'

        # PRIORITY 3: Greeks deterioration

        # Vega/Theta ratio
        vt_ratio = abs(current_greeks['vega'] / current_greeks['theta']) if current_greeks['theta'] != 0 else 999
        if vt_ratio < self.greeks_config['vega_theta_ratio_min']:
            return True, f'vt_ratio_{vt_ratio:.2f}'

        # Vega decline
        vega_decline = (position.entry_vega - current_greeks[
            'vega']) / position.entry_vega if position.entry_vega > 0 else 0
        if vega_decline > self.greeks_config['vega_decline_pct']:
            return True, f'vega_decline_{vega_decline:.2%}'

        return False, None

    def _execute_straddle(self, trade_date: datetime, options: pl.DataFrame,
                          predicted_vol: float, signal_strength: float,
                          horizon_days: int) -> Optional[Position]:
        """Execute ATM straddle (LONG only)"""
        S = options['active_underlying_price'][0]

        calls = options.filter(pl.col('option_type').str.to_lowercase() == 'c')
        puts = options.filter(pl.col('option_type').str.to_lowercase() == 'p')

        if len(calls) == 0 or len(puts) == 0:
            return None

        call_strikes = calls['strike'].to_numpy()
        atm_strike = call_strikes[np.argmin(np.abs(call_strikes - S))]

        call_option = calls.filter(pl.col('strike') == atm_strike)
        put_option = puts.filter(pl.col('strike') == atm_strike)

        if len(call_option) == 0 or len(put_option) == 0:
            return None

        call_option = call_option[0]
        put_option = put_option[0]

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

        # Calculate Greeks at entry
        call_greeks = self._black_scholes_greeks(S, atm_strike, T, r, call_iv, 'c')
        put_greeks = self._black_scholes_greeks(S, atm_strike, T, r, put_iv, 'p')
        avg_vega = (call_greeks['vega'] + put_greeks['vega']) / 2
        avg_theta = (call_greeks['theta'] + put_greeks['theta']) / 2

        position_multiplier = self._calculate_position_multiplier(signal_strength)
        position_size = self.base_position_size * position_multiplier

        call_premium = call_option['mid'][0]
        put_premium = put_option['mid'][0]
        total_premium = call_premium + put_premium

        n_contracts = position_size / (total_premium * 100)

        if n_contracts < 0.01:
            return None

        position_id = f"straddle_{trade_date.strftime('%Y%m%d')}_{atm_strike:.0f}"

        return Position(
            position_id=position_id,
            entry_date=trade_date.date(),
            entry_time=call_option['quote_datetime'][0],
            strategy='straddle',
            call_strike=atm_strike,
            put_strike=atm_strike,
            expiration=call_option['expiration'][0],
            days_to_expiry=horizon_days,
            n_contracts=n_contracts,
            entry_iv=avg_iv,
            entry_vega=avg_vega,
            entry_theta=avg_theta,
            entry_premium=total_premium,
            entry_underlying=S,
            predicted_vol=predicted_vol,
            entry_signal=signal_strength
        )

    def _execute_strangle(self, trade_date: datetime, options: pl.DataFrame,
                          predicted_vol: float, signal_strength: float,
                          horizon_days: int) -> Optional[Position]:
        """Execute OTM strangle (LONG only)"""
        S = options['active_underlying_price'][0]

        calls = options.filter(pl.col('option_type').str.to_lowercase() == 'c')
        puts = options.filter(pl.col('option_type').str.to_lowercase() == 'p')

        if len(calls) == 0 or len(puts) == 0:
            return None

        put_target_strike = S * self.strangle_strikes[0]
        call_target_strike = S * self.strangle_strikes[1]

        put_strikes = puts['strike'].to_numpy()
        call_strikes = calls['strike'].to_numpy()

        put_strike = put_strikes[np.argmin(np.abs(put_strikes - put_target_strike))]
        call_strike = call_strikes[np.argmin(np.abs(call_strikes - call_target_strike))]

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

        # Calculate Greeks at entry
        call_greeks = self._black_scholes_greeks(S, call_strike, T, r, call_iv, 'c')
        put_greeks = self._black_scholes_greeks(S, put_strike, T, r, put_iv, 'p')
        avg_vega = (call_greeks['vega'] + put_greeks['vega']) / 2
        avg_theta = (call_greeks['theta'] + put_greeks['theta']) / 2

        position_multiplier = self._calculate_position_multiplier(signal_strength)
        position_size = self.base_position_size * position_multiplier

        total_premium = call_premium + put_premium
        n_contracts = position_size / (total_premium * 100)

        if n_contracts < 0.01:
            return None

        position_id = f"strangle_{trade_date.strftime('%Y%m%d')}_{call_strike:.0f}_{put_strike:.0f}"

        return Position(
            position_id=position_id,
            entry_date=trade_date.date(),
            entry_time=call_option['quote_datetime'][0],
            strategy='strangle',
            call_strike=call_strike,
            put_strike=put_strike,
            expiration=call_option['expiration'][0],
            days_to_expiry=horizon_days,
            n_contracts=n_contracts,
            entry_iv=avg_iv,
            entry_vega=avg_vega,
            entry_theta=avg_theta,
            entry_premium=total_premium,
            entry_underlying=S,
            predicted_vol=predicted_vol,
            entry_signal=signal_strength
        )

    def _calculate_position_multiplier(self, signal_strength: float) -> float:
        """Calculate position size multiplier based on signal strength."""
        signal_excess = signal_strength - self.entry_threshold

        if abs(self.entry_threshold) > 1e-6:
            position_multiplier = signal_excess / abs(self.entry_threshold)
        else:
            position_multiplier = abs(signal_excess) / 0.01

        position_multiplier = min(max(position_multiplier, 0), self.signal_multiplier)

        return position_multiplier

    def _get_options_for_date(self, trade_date: datetime, days_to_expiry: int) -> Optional[pl.DataFrame]:
        """Get options for entry scanning (first quote of day)."""
        date_key = trade_date.date()

        date_options = self.entry_scan_df.filter(pl.col('quote_date') == date_key)

        if len(date_options) == 0:
            return None

        # Find options with target maturity
        target_options = date_options.filter(
            (pl.col('days_to_expiry') >= days_to_expiry - 2) &
            (pl.col('days_to_expiry') <= days_to_expiry + 2)
        )

        if len(target_options) > 0:
            return target_options

        return None

    def _calculate_market_iv(self, options: pl.DataFrame, trade_date: datetime,
                             horizon_days: int) -> Optional[float]:
        """Calculate average market implied volatility"""
        if options is None or len(options) == 0:
            return None

        S = options['active_underlying_price'][0]
        T = horizon_days / 365.0
        r = self._get_risk_free_rate(trade_date)

        prices = options['mid'].to_numpy()
        strikes = options['strike'].to_numpy()
        option_types = options['option_type'].to_numpy()

        ivs = self._implied_volatility_vectorized(prices, S, strikes, T, r, option_types)
        valid_ivs = ivs[~np.isnan(ivs)]

        if len(valid_ivs) == 0:
            return None

        return float(np.mean(valid_ivs))

    def _should_trade_long(self, predicted_vol: float, market_iv: float) -> bool:
        """Check if we should enter a LONG volatility trade."""
        signal_strength = predicted_vol - market_iv
        return signal_strength > self.entry_threshold

    def _get_expiration_price(self, expiration_date: datetime) -> Optional[float]:
        """Get underlying price at expiration."""
        exp_date = expiration_date.date() if isinstance(expiration_date, datetime) else expiration_date
        return self.expiration_prices.get(exp_date, None)

    def _calculate_payoff(self, position: Position, expiration_price: float) -> Dict:
        """Calculate P&L for LONG options positions at expiration."""
        call_strike = position.call_strike
        put_strike = position.put_strike
        n_contracts = position.n_contracts

        call_intrinsic = max(expiration_price - call_strike, 0)
        put_intrinsic = max(put_strike - expiration_price, 0)

        call_value = call_intrinsic * 100 * n_contracts
        put_value = put_intrinsic * 100 * n_contracts
        total_intrinsic_value = call_value + put_value

        total_premium_value = position.entry_premium * 100 * n_contracts

        net_pnl = total_intrinsic_value - total_premium_value
        return_pct = (net_pnl / total_premium_value) * 100 if total_premium_value > 0 else 0

        return {
            'position_id': position.position_id,
            'entry_date': position.entry_date,
            'entry_time': position.entry_time,
            'exit_date': position.expiration,
            'exit_time': position.expiration,
            'exit_reason': 'expiration',
            'strategy': position.strategy,
            'call_strike': call_strike,
            'put_strike': put_strike,
            'expiration': position.expiration,
            'days_held': (position.expiration.date() - position.entry_date).days,
            'n_contracts': n_contracts,
            'entry_premium': position.entry_premium,
            'entry_underlying': position.entry_underlying,
            'exit_underlying': expiration_price,
            'entry_iv': position.entry_iv,
            'exit_iv': None,
            'entry_signal': position.entry_signal,
            'exit_signal': None,
            'entry_vega': position.entry_vega,
            'exit_vega': None,
            'entry_theta': position.entry_theta,
            'exit_theta': None,
            'peak_iv': position.peak_iv,
            'peak_pnl': position.peak_pnl,
            'predicted_vol': position.predicted_vol,
            'call_intrinsic': call_intrinsic,
            'put_intrinsic': put_intrinsic,
            'total_intrinsic_value': total_intrinsic_value,
            'total_premium_value': total_premium_value,
            'net_pnl': net_pnl,
            'return_pct': return_pct
        }

    def run_backtest(self) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """
        Run the backtest with signal reversal and Greeks exits.
        OPTIMIZED FOR COMPUTE SPEED.

        Returns
        -------
        trades_df : pl.DataFrame
            DataFrame with all completed trades
        equity_df : pl.DataFrame
            DataFrame with equity curve over time
        tracking_df : pl.DataFrame
            DataFrame with position tracking history
        """
        print("\n" + "=" * 80)
        print("STARTING BACKTEST V2 - WITH EXITS")
        print("=" * 80)
        print(f"Strategy: {self.strategy_type}")
        print(f"Entry threshold: {self.entry_threshold:.3%}")
        print(f"Exit threshold: {self.exit_threshold:.3%}")
        print(f"Tracking frequency: {self.tracking_freq}")
        print(f"Stop loss: {self.stop_loss_pct:.1%}")
        print(f"Greeks config: {self.greeks_config}")
        print("-" * 80)

        start_time = time.time()

        current_capital = self.initial_capital
        self.equity_curve = [{
            'date': self.predictions_df['date'].min(),
            'equity': current_capital
        }]

        maturity_days_map = {'1W': 5, '2W': 10, '1M': 20, '1Q': 60}

        if self.maturity_filter:
            maturities_to_trade = {
                self.maturity_filter: maturity_days_map[self.maturity_filter]
            }
        else:
            maturities_to_trade = maturity_days_map

        trading_dates = sorted(self.predictions_df['date'].unique().to_list())

        # Get all tracking timestamps
        tracking_timestamps = sorted(list(set([k[0] for k in self.options_by_datetime.keys()])))

        print(f"Trading dates: {len(trading_dates)}")
        print(f"Tracking timestamps: {len(tracking_timestamps)}")
        print(f"Approx checks per day: {len(tracking_timestamps) / len(trading_dates):.1f}")

        positions_opened = 0
        positions_closed = 0

        # MAIN LOOP: Iterate through all tracking timestamps
        for idx, current_time in enumerate(tracking_timestamps):
            current_date = current_time.date()

            if idx % 500 == 0:
                print(f"Progress: {idx}/{len(tracking_timestamps)} ({idx / len(tracking_timestamps) * 100:.1f}%) - "
                      f"Open positions: {len(self.open_positions)}, Closed: {positions_closed}")

            # Check exits for all open positions
            positions_to_close = []

            for pos_id, position in self.open_positions.items():
                # Get current option prices
                call_data = self._get_option_price(
                    current_time, position.expiration,
                    position.call_strike, 'C'
                )
                put_data = self._get_option_price(
                    current_time, position.expiration,
                    position.put_strike, 'P'
                )

                if call_data is None or put_data is None:
                    continue

                # Calculate current Greeks and P&L
                current_prices = {
                    'timestamp': current_time,
                    'underlying': call_data['underlying'],
                    'days_to_expiry': call_data['days_to_expiry']
                }

                current_greeks = self._calculate_position_greeks(
                    position, current_time, current_prices
                )

                if current_greeks is None:
                    continue

                current_greeks['days_to_expiry'] = call_data['days_to_expiry']

                current_pnl = self._calculate_position_pnl(position, current_prices)

                # Update peak tracking
                position.update_peaks(current_time, current_greeks['avg_iv'], current_pnl)

                # Check exit conditions
                should_exit, exit_reason = self._check_exit_conditions(
                    position, current_time, current_greeks, current_pnl
                )

                if should_exit:
                    # Close position
                    position.exit_date = current_date
                    position.exit_time = current_time
                    position.exit_reason = exit_reason
                    position.exit_iv = current_greeks['avg_iv']
                    position.exit_vega = current_greeks['vega']
                    position.exit_theta = current_greeks['theta']
                    position.exit_pnl = current_pnl
                    position.exit_underlying = current_prices['underlying']
                    position.status = 'closed'

                    positions_to_close.append(pos_id)

                    # Update capital
                    current_capital += current_pnl
                    self.equity_curve.append({
                        'date': current_time,
                        'equity': current_capital,
                        'trade_pnl': current_pnl
                    })

                    positions_closed += 1

            # Remove closed positions
            for pos_id in positions_to_close:
                self.closed_positions.append(self.open_positions[pos_id])
                del self.open_positions[pos_id]

            # Check for new entries (only at first quote of each day)
            if current_time.hour == 9 and current_time.minute == 30:
                trade_date = datetime.combine(current_date, datetime.min.time())

                for maturity_name, horizon_days in maturities_to_trade.items():
                    # Check if we have predictions for this date-horizon
                    pred_key = (trade_date, horizon_days)
                    if pred_key not in self.predictions_lookup:
                        continue

                    pred_data = self.predictions_lookup[pred_key]
                    predicted_vol = pred_data['predicted_vol']

                    # Get options for entry
                    options = self._get_options_for_date(trade_date, horizon_days)

                    if options is None or len(options) == 0:
                        continue

                    market_iv = self._calculate_market_iv(options, trade_date, horizon_days)

                    if market_iv is None:
                        continue

                    signal_strength = predicted_vol - market_iv

                    should_trade = self._should_trade_long(predicted_vol, market_iv)

                    if not should_trade:
                        continue

                    # Execute trades
                    if self.strategy_type in ['straddle', 'both']:
                        position = self._execute_straddle(
                            trade_date, options, predicted_vol,
                            signal_strength, horizon_days
                        )
                        if position:
                            self.open_positions[position.position_id] = position
                            positions_opened += 1

                    if self.strategy_type in ['strangle', 'both']:
                        position = self._execute_strangle(
                            trade_date, options, predicted_vol,
                            signal_strength, horizon_days
                        )
                        if position:
                            self.open_positions[position.position_id] = position
                            positions_opened += 1

        # Close any remaining open positions at expiration
        for pos_id, position in list(self.open_positions.items()):
            expiration_price = self._get_expiration_price(position.expiration)

            if expiration_price is None:
                continue

            payoff = self._calculate_payoff(position, expiration_price)
            self.closed_positions.append({
                **payoff,
                'peak_iv': position.peak_iv,
                'peak_pnl': position.peak_pnl
            })

            current_capital += payoff['net_pnl']
            self.equity_curve.append({
                'date': position.expiration,
                'equity': current_capital,
                'trade_pnl': payoff['net_pnl']
            })

        runtime = time.time() - start_time

        print(f"\n{'=' * 80}")
        print("BACKTEST COMPLETE")
        print(f"{'=' * 80}")
        print(f"Runtime: {runtime:.2f} seconds")
        print(f"Positions opened: {positions_opened}")
        print(f"Positions closed: {positions_closed}")
        print(f"Positions closed early: {len([p for p in self.closed_positions if isinstance(p, Position)])}")
        print(
            f"Positions held to expiration: {positions_closed - len([p for p in self.closed_positions if isinstance(p, Position)])}")
        print(f"{'=' * 80}\n")

        # Convert to DataFrames
        if len(self.closed_positions) > 0:
            # Convert Position objects to dicts
            trades_data = []
            for pos in self.closed_positions:
                if isinstance(pos, Position):
                    entry_value = pos.entry_premium * 100 * pos.n_contracts
                    trades_data.append({
                        'position_id': pos.position_id,
                        'entry_date': pos.entry_date,
                        'entry_time': pos.entry_time,
                        'exit_date': pos.exit_date,
                        'exit_time': pos.exit_time,
                        'exit_reason': pos.exit_reason,
                        'strategy': pos.strategy,
                        'call_strike': pos.call_strike,
                        'put_strike': pos.put_strike,
                        'expiration': pos.expiration,
                        'days_held': (pos.exit_date - pos.entry_date).days,
                        'n_contracts': pos.n_contracts,
                        'entry_premium': pos.entry_premium,
                        'entry_underlying': pos.entry_underlying,
                        'exit_underlying': pos.exit_underlying,
                        'entry_iv': pos.entry_iv,
                        'exit_iv': pos.exit_iv,
                        'entry_signal': pos.entry_signal,
                        'exit_signal': pos.predicted_vol - pos.exit_iv if pos.exit_iv else None,
                        'entry_vega': pos.entry_vega,
                        'exit_vega': pos.exit_vega,
                        'entry_theta': pos.entry_theta,
                        'exit_theta': pos.exit_theta,
                        'peak_iv': pos.peak_iv,
                        'peak_pnl': pos.peak_pnl,
                        'predicted_vol': pos.predicted_vol,
                        'net_pnl': pos.exit_pnl,
                        'return_pct': (pos.exit_pnl / entry_value * 100) if entry_value > 0 else 0,
                        'total_premium_value': entry_value
                    })
                else:
                    trades_data.append(pos)

            trades_df = pl.DataFrame(trades_data)
        else:
            trades_df = pl.DataFrame()

        equity_df = pl.DataFrame(self.equity_curve).sort('date')

        # Save results
        if len(trades_df) > 0:
            suffix = f"{self.tracking_freq}_{self.maturity_filter or 'all'}"
            trades_path = self.output_dir / f'trades_v2_{suffix}.csv'
            equity_path = self.output_dir / f'equity_v2_{suffix}.csv'

            trades_df.write_csv(str(trades_path))
            equity_df.write_csv(str(equity_path))

            print(f"✓ Saved trades to: {trades_path}")
            print(f"✓ Saved equity curve to: {equity_path}\n")

        self._print_summary(trades_df, current_capital, runtime)

        return trades_df, equity_df, pl.DataFrame()

    def _print_summary(self, trades_df: pl.DataFrame, final_capital: float, runtime: float):
        """Print backtest summary statistics with Sortino ratio."""
        if len(trades_df) == 0:
            print("\nNo trades executed!")
            return

        net_pnls = trades_df['net_pnl'].to_numpy()
        return_pcts = trades_df['return_pct'].to_numpy()

        winners = np.sum(net_pnls > 0)
        losers = np.sum(net_pnls < 0)

        print("=" * 80)
        print("BACKTEST RESULTS")
        print("=" * 80)
        print(f"Runtime: {runtime:.2f} seconds")
        print(f"Tracking frequency: {self.tracking_freq}")
        print(f"\nCapital:")
        print(f"  Initial: ${self.initial_capital:,.2f}")
        print(f"  Final: ${final_capital:,.2f}")
        print(f"  Total P&L: ${final_capital - self.initial_capital:,.2f}")
        print(f"  Return: {((final_capital / self.initial_capital) - 1) * 100:.2f}%")

        print(f"\nTrades:")
        print(f"  Total: {len(trades_df)}")
        print(f"  Winners: {winners} ({winners / len(trades_df) * 100:.1f}%)")
        print(f"  Losers: {losers} ({losers / len(trades_df) * 100:.1f}%)")

        print(f"\nP&L Statistics:")
        print(f"  Avg P&L per trade: ${np.mean(net_pnls):,.2f}")
        print(f"  Median P&L: ${np.median(net_pnls):,.2f}")
        print(f"  Best trade: ${np.max(net_pnls):,.2f}")
        print(f"  Worst trade: ${np.min(net_pnls):,.2f}")

        print(f"\nReturn Statistics:")
        print(f"  Avg return: {np.mean(return_pcts):.2f}%")
        print(f"  Median return: {np.median(return_pcts):.2f}%")
        print(f"  Std dev: {np.std(return_pcts):.2f}%")

        if np.std(return_pcts) > 0:
            sharpe = np.mean(return_pcts) / np.std(return_pcts)
            print(f"  Sharpe ratio: {sharpe:.3f}")

        # SORTINO RATIO (downside deviation only)
        negative_returns = return_pcts[return_pcts < 0]
        if len(negative_returns) > 0:
            downside_std = np.sqrt(np.mean(negative_returns ** 2))
            sortino = np.mean(return_pcts) / downside_std if downside_std > 0 else 0
            print(f"  Sortino ratio: {sortino:.3f} ⭐")

        # Exit reason analysis
        if 'exit_reason' in trades_df.columns:
            print(f"\nExit Reasons:")
            exit_counts = trades_df.group_by('exit_reason').agg(pl.count()).sort('count', descending=True)
            for row in exit_counts.iter_rows(named=True):
                pct = row['count'] / len(trades_df) * 100
                print(f"  {row['exit_reason']}: {row['count']} ({pct:.1f}%)")

        # Hold period analysis
        if 'days_held' in trades_df.columns:
            print(f"\nHold Period:")
            print(f"  Avg: {trades_df['days_held'].mean():.1f} days")
            print(f"  Median: {trades_df['days_held'].median():.1f} days")
            print(f"  Min: {trades_df['days_held'].min()} days")
            print(f"  Max: {trades_df['days_held'].max()} days")

        print("=" * 80)