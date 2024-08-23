import os
import datetime
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
    def __init__(self, data, lookback_period, model):
        self.model = model
        print(f"Using pretrained model: {model}")
        self.data = data
        self.lookback_period = lookback_period
        self.asset = 'EQ:SPY'
        self.signals = pd.Series(0.0, index=data.index)
        super().__init__({self.asset: self.signals})
        self.current_position = 0  # 0 for no position, >0 for long, <0 for short

    def __call__(self, dt):
        print(f"Predicting for {dt}...")
        end_date = pd.to_datetime(dt).tz_convert(self.data.index.tz)

        loc = self.data.index.get_loc(self.data.index.asof(end_date))
        start_date = self.data.index[loc - self.lookback_period]

        historical_data = self.data.loc[start_date:end_date]

        # Move Date index to a column
        historical_data.reset_index(inplace=True)

        prediction = predict(self.model, [historical_data], device="cpu")
        print(f"Prediction for {dt}: {prediction}")

        # Ensure prediction is a single scalar value
        if isinstance(prediction, (np.ndarray, list)):
            prediction = np.mean(prediction)

        # Clip the prediction to ensure it's between -1 and 1
        prediction = np.clip(prediction, -1, 1)

        # Implement the strategy
        if prediction > 0:
            if self.current_position <= 0:
                self.current_position = prediction
                return {self.asset: prediction}  # Go long
        elif prediction < 0:
            if self.current_position >= 0:
                self.current_position = prediction
                return {self.asset: prediction}  # Go short
        else:  # signal == 0
            if self.current_position != 0:
                self.current_position = 0
                return {self.asset: 0.0}  # Close position

        # If no action needed, return current position
        return {self.asset: float(self.current_position)}


def load_data(csv_file):
    """
    Load the S&P 500 data from a CSV file.
    """
    return pd.read_csv(csv_file, index_col='Date', parse_dates=True)


def backtest(csv_file, initial_cash, lookback_period,
             start_dt="2019-06-30 14:30:00",
             end_dt="2024-06-30 23:59:00",
             model="18989.US.RANDOM.30.5.64.3.RANGE.2.49"):
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
    strategy_alpha_model = PredictiveAlphaModel(data, lookback_period, model)

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
    results = strategy.run()

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
    benchmark_results = benchmark_backtest.run()

    # Generate and save tearsheet
    tearsheet = TearsheetStatistics(
        strategy_equity=strategy.get_equity_curve(),
        benchmark_equity=benchmark_backtest.get_equity_curve(),
        title='Predictive Strategy'
    )
    tearsheet.plot_results(f'results/predictive_strategy_backtest_{datetime.datetime.now()}.png')

    return results, tearsheet
