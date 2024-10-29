import os
from deep_learning import predict
import pandas as pd
import numpy as np
import pytz
from qstrader import settings
from qstrader.alpha_model.fixed_signals import FixedSignalsAlphaModel
from qstrader.asset.equity import Equity
from qstrader.asset.universe.static import StaticUniverse
from qstrader.data.backtest_data_handler import BacktestDataHandler
from qstrader.data.daily_bar_csv import CSVDailyBarDataSource
from qstrader.statistics.tearsheet import TearsheetStatistics
from qstrader.trading.backtest import BacktestTradingSession
from qstrader.broker.fee_model.percent_fee_model import PercentFeeModel


__all__ = ['backtest']


class PredictiveAlphaModel(FixedSignalsAlphaModel):
    def __init__(self, data, lookback_period, model, initial_cash, risk_per_trade=0.01):
        self.model = model
        print(f"Using pretrained model: {model}")
        self.data = data
        self.lookback_period = lookback_period
        self.asset = 'EQ:SPY'
        self.signals = pd.Series(0.0, index=data.index)
        super().__init__({self.asset: self.signals})
        self.current_position = 0  # 0 for no position, 1 for long, -1 for short
        self.entry_price = None
        self.initial_cash = initial_cash
        self.risk_per_trade = risk_per_trade  # Risk % of capital per trade
        self.max_position_size = 0.4  # Max 40% of portfolio per position

    def __call__(self, dt):
        print(f"Predicting for {dt}...")
        end_date = pd.to_datetime(dt).tz_convert(self.data.index.tz)

        # Align end_date with available data
        asof_date = self.data.index.asof(end_date)
        if pd.isna(asof_date):
            # No data available before end_date
            return {self.asset: 0.0}

        try:
            loc = self.data.index.get_loc(asof_date)
            if loc < self.lookback_period:
                return {self.asset: 0.0}
            start_date = self.data.index[loc - self.lookback_period]
            historical_data = self.data.loc[start_date:asof_date]
        except KeyError:
            return {self.asset: 0.0}

        # Prepare data for prediction
        historical_data_reset = historical_data.reset_index()

        # Generate prediction
        prediction = predict(self.model, [historical_data_reset], device="cpu")
        print(f"Prediction for {dt}: {prediction}")

        # Ensure prediction is a single scalar value
        if isinstance(prediction, (np.ndarray, list)):
            prediction = np.mean(prediction)

        # Clip the prediction to ensure it's between -1 and 1
        prediction = np.clip(prediction, -1, 1)

        # Calculate technical indicators
        close_prices = historical_data['Close']

        # MACD
        exp1 = close_prices.ewm(span=12, adjust=False).mean()
        exp2 = close_prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=9, adjust=False).mean()
        macd_signal = macd.iloc[-1] - signal_line.iloc[-1]

        # RSI
        delta = close_prices.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        avg_gain = up.rolling(window=14).mean()
        avg_loss = down.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        latest_rsi = rsi.iloc[-1]

        # Bollinger Bands
        sma = close_prices.rolling(window=20).mean()
        std = close_prices.rolling(window=20).std()
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        latest_close = close_prices.iloc[-1]
        upper_band_value = upper_band.iloc[-1]
        lower_band_value = lower_band.iloc[-1]

        # Simple Moving Average
        sma_50 = close_prices.rolling(window=50).mean().iloc[-1]
        sma_200 = close_prices.rolling(window=200).mean().iloc[-1]

        # Determine technical signal
        technical_signal = 0
        if (macd_signal > 0) and (latest_rsi > 50) and (latest_close > sma_50 > sma_200):
            technical_signal = 1  # Bullish signal
        elif (macd_signal < 0) and (latest_rsi < 50) and (latest_close < sma_50 < sma_200):
            technical_signal = -1  # Bearish signal

        # Combine prediction and technical signal
        combined_signal = prediction + technical_signal

        # Risk Management Parameters
        # Use ATR for dynamic stop-loss and take-profit
        high_prices = historical_data['High']
        low_prices = historical_data['Low']
        atr = (high_prices - low_prices).rolling(window=14).mean().iloc[-1]
        stop_loss_pct = (atr / latest_close) * 1  # Stop-loss at 1.5 times ATR
        take_profit_pct = (atr / latest_close) * 2  # Take-profit at 3 times ATR

        signal = 0.0

        # Implement the strategy based on combined signal
        if combined_signal > 0:
            if self.current_position <= 0:
                # Calculate position size based on risk per trade
                risk_amount = self.initial_cash * self.risk_per_trade
                # Stop-loss price for long position
                stop_loss_price = latest_close - (stop_loss_pct * latest_close)
                # Number of shares to buy
                shares = risk_amount / (latest_close - stop_loss_price)
                if not shares or np.isnan(shares):
                    shares = risk_amount / latest_close
                # Position size as a fraction of total capital
                position_value = shares * latest_close
                print(f"Position value: {position_value}")
                position_pct = min(position_value / self.initial_cash, self.max_position_size)
                self.current_position = position_pct
                self.entry_price = latest_close
                print(f"Entering long position at {latest_close} with {position_pct * 100:.2f}% of portfolio")
                return {self.asset: position_pct}  # Go long
        elif combined_signal < 0:
            if self.current_position >= 0:
                # Calculate position size based on risk per trade
                risk_amount = self.initial_cash * self.risk_per_trade
                # Stop-loss price for short position
                stop_loss_price = latest_close + (stop_loss_pct * latest_close)
                # Number of shares to short
                shares = risk_amount / (stop_loss_price - latest_close)
                if not shares or np.isnan(shares):
                    shares = risk_amount / latest_close
                # Position size as a fraction of total capital (negative for short)
                position_value = shares * latest_close
                print(f"Position value: {position_value}")
                position_pct = -min(position_value / self.initial_cash, self.max_position_size)
                self.current_position = position_pct
                self.entry_price = latest_close
                print(f"Entering short position at {latest_close} with {abs(position_pct) * 100:.2f}% of portfolio")
                return {self.asset: position_pct}  # Go short
        else:  # combined_signal == 0
            if self.current_position != 0:
                self.current_position = 0
                self.entry_price = None
                print(f"Exiting position at {dt}")
                return {self.asset: 0.0}  # Close position

        # Check for stop-loss or take-profit conditions
        if self.current_position != 0 and self.entry_price is not None:
            if self.current_position > 0:
                # Long position
                if latest_close <= self.entry_price - (stop_loss_pct * self.entry_price):
                    print(f"Stop-loss triggered for long position at {latest_close}")
                    self.current_position = 0
                    self.entry_price = None
                    return {self.asset: 0.0}  # Close position
                elif latest_close >= self.entry_price + (take_profit_pct * self.entry_price):
                    print(f"Take-profit triggered for long position at {latest_close}")
                    self.current_position = 0
                    self.entry_price = None
                    return {self.asset: 0.0}  # Close position
            else:
                # Short position
                if latest_close >= self.entry_price + (stop_loss_pct * self.entry_price):
                    print(f"Stop-loss triggered for short position at {latest_close}")
                    self.current_position = 0
                    self.entry_price = None
                    return {self.asset: 0.0}  # Close position
                elif latest_close <= self.entry_price - (take_profit_pct * self.entry_price):
                    print(f"Take-profit triggered for short position at {latest_close}")
                    self.current_position = 0
                    self.entry_price = None
                    return {self.asset: 0.0}  # Close position

        # Maintain current position if no action is needed
        return {self.asset: float(self.current_position)}



def load_data(csv_file):
    """
    Load the S&P 500 data from a CSV file.
    """
    return pd.read_csv(csv_file, index_col='Date', parse_dates=True)


def backtest(csv_file, initial_cash, lookback_period,
             start_dt="2019-06-30 14:30:00",
             end_dt="2024-06-30 23:59:00",
             model="18989.US.RANDOM.30.5.64.3.RANGE.2.49",
             save_dir=None) -> TearsheetStatistics:
    # Load the data
    data = load_data(csv_file)

    start_dt = pd.Timestamp(start_dt, tz=pytz.UTC)
    end_dt = pd.Timestamp(end_dt, tz=pytz.UTC)

    # Construct the symbols and assets necessary for the backtest
    strategy_symbols = ['SPY']
    strategy_assets = ['EQ:SPY']
    strategy_universe = StaticUniverse(strategy_assets)
    strategy_data_source = CSVDailyBarDataSource(os.path.join(os.path.abspath('.'), 'data', 'unprocessed'), Equity, csv_symbols=strategy_symbols)
    strategy_data_handler = BacktestDataHandler(strategy_universe,
                                                data_sources=[strategy_data_source])

    # Create the alpha model
    strategy_alpha_model = PredictiveAlphaModel(
        data,
        lookback_period,
        model,
        initial_cash=initial_cash,
        risk_per_trade=0.10  # Risk 5% of capital per trade
    )
    # Set up the backtest trading session
    strategy = BacktestTradingSession(
        start_dt=start_dt,
        end_dt=end_dt,
        universe=strategy_universe,
        alpha_model=strategy_alpha_model,
        portfolio_id='predictive_strategy',
        initial_cash=initial_cash,
        gross_leverage=1.0,
        rebalance='weekly',
        rebalance_weekday='FRI',
        rebalance_calendar='NYSE',
        data_handler=strategy_data_handler,
        fee_model=PercentFeeModel(0.001)
    )

    # Run the backtest
    strategy.run()

    # Construct benchmark assets (buy & hold SPY)
    benchmark_symbols = ['SPY']
    benchmark_assets = ['EQ:SPY']
    benchmark_universe = StaticUniverse(benchmark_assets)
    benchmark_data_source = CSVDailyBarDataSource(os.path.join(os.path.abspath('.'), 'data', 'unprocessed'), Equity, csv_symbols=benchmark_symbols)
    benchmark_data_handler = BacktestDataHandler(benchmark_universe,
                                                 data_sources=[benchmark_data_source])

    # Construct a benchmark Alpha Model that provides
    # 100% static allocation to the SPY ETF, with no rebalance
    benchmark_alpha_model = FixedSignalsAlphaModel({'EQ:SPY': 1.0})
    benchmark_backtest = BacktestTradingSession(
        start_dt,
        end_dt,
        benchmark_universe,
        benchmark_alpha_model,
        rebalance='buy_and_hold',
        long_only=True,
        cash_buffer_percentage=0.01,
        data_handler=benchmark_data_handler,
        initial_cash=initial_cash,
        fee_model=PercentFeeModel(0.001)
    )
    benchmark_backtest.run()

    # Generate and save tearsheet
    tearsheet = TearsheetStatistics(
        strategy_equity=strategy.get_equity_curve(),
        benchmark_equity=benchmark_backtest.get_equity_curve(),
        title='Predictive Strategy'
    )
    tearsheet.plot_results(filename=save_dir)

    return tearsheet
