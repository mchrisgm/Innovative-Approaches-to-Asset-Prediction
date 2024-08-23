import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import qstrader.statistics.performance as perf
import seaborn as sns


__all__ = ['calculate_beta', 'calculate_alpha',
           'calculate_information_ratio', 'calculate_metrics',
           'monthly_returns', 'weekly_returns',
           'distribution']


def calculate_beta(strategy_returns, benchmark_returns):
    """
    Calculate the beta of a strategy relative to a benchmark.

    :param strategy_returns: numpy array of strategy returns
    :param benchmark_returns: numpy array of benchmark returns
    :return: beta value
    """
    covariance = np.cov(strategy_returns, benchmark_returns)[0, 1]
    benchmark_variance = np.var(benchmark_returns)
    return covariance / benchmark_variance


def calculate_alpha(strategy_returns, benchmark_returns, risk_free_rate, beta):
    """
    Calculate the alpha of a strategy.

    :param strategy_returns: numpy array of strategy returns
    :param benchmark_returns: numpy array of benchmark returns
    :param risk_free_rate: risk-free rate (as a decimal, e.g., 0.02 for 2%)
    :param beta: beta of the strategy relative to the benchmark
    :return: alpha value
    """
    strategy_return = np.mean(strategy_returns)
    benchmark_return = np.mean(benchmark_returns)
    return strategy_return - (risk_free_rate + beta * (benchmark_return - risk_free_rate))


def calculate_information_ratio(strategy_returns, benchmark_returns):
    """
    Calculate the information ratio of a strategy.

    :param strategy_returns: numpy array of strategy returns
    :param benchmark_returns: numpy array of benchmark returns
    :return: information ratio
    """
    excess_returns = strategy_returns - benchmark_returns
    tracking_error = np.std(excess_returns)
    return np.mean(excess_returns) / tracking_error


def calculate_metrics(strategy_returns, benchmark_returns, risk_free_rate):
    """
    Calculate beta, alpha, and information ratio for a strategy.

    :param strategy_returns: numpy array of strategy returns
    :param benchmark_returns: numpy array of benchmark returns
    :param risk_free_rate: risk-free rate (as a decimal, e.g., 0.02 for 2%)
    :return: dictionary containing beta, alpha, and information ratio
    """
    beta = calculate_beta(strategy_returns, benchmark_returns)
    alpha = calculate_alpha(strategy_returns, benchmark_returns, risk_free_rate, beta) * 252
    information_ratio = calculate_information_ratio(strategy_returns, benchmark_returns) * np.sqrt(252)

    return {
        'beta': beta,
        'alpha': alpha,
        'information_ratio': information_ratio
    }


def monthly_returns(stats, ax=None, **kwargs):
        """
        Plots a heatmap of the monthly returns.
        """
        returns = stats['returns']
        if ax is None:
            ax = plt.gca()

        monthly_ret = perf.aggregate_returns(returns, 'monthly')
        monthly_ret = monthly_ret.unstack()
        monthly_ret = np.round(monthly_ret, 3)
        monthly_ret.rename(
            columns={1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr',
                     5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug',
                     9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'},
            inplace=True
        )

        sns.heatmap(
            monthly_ret.fillna(0) * 100.0,
            annot=True,
            fmt="0.1f",
            annot_kws={"size": 8},
            alpha=1.0,
            center=0.0,
            cbar=False,
            cmap=cm.RdYlGn,
            ax=ax, **kwargs)
        ax.set_title('Monthly Returns (%)', fontweight='bold')
        ax.set_ylabel('')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_xlabel('')

        return monthly_ret


def weekly_returns(stats, ax=None, **kwargs):
        """
        Plots a heatmap of the monthly returns.
        """
        returns = stats['returns']
        if ax is None:
            ax = plt.gca()

        weekly_ret = perf.aggregate_returns(returns, 'weekly')
        weekly_ret = weekly_ret.unstack()
        weekly_ret = np.round(weekly_ret, 3)

        sns.heatmap(
            weekly_ret.fillna(0) * 100.0,
            annot=True,
            fmt="0.1f",
            annot_kws={"size": 8},
            alpha=1.0,
            center=0.0,
            cbar=False,
            cmap=cm.RdYlGn,
            ax=ax, **kwargs)
        ax.set_title('Monthly Returns (%)', fontweight='bold')
        ax.set_ylabel('')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_xlabel('')

        return weekly_ret


def distribution(df):
    df = df.copy()
    df = df * 100.0
    # Flatten the DataFrame to a 1D array
    data = df.values.flatten()

    # Drop any NaN values (if present)
    data = data[~np.isnan(data)]

    # Calculate mean and standard deviation
    mean = np.mean(data)
    std = np.std(data)

    # Plot the distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data, kde=True, color='skyblue', stat='density', linewidth=0)

    # Add lines for mean and standard deviation
    plt.axvline(mean, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean:.2f}')
    plt.axvline(mean + std, color='green', linestyle='dashed', linewidth=2, label=f'+1 Std Dev: {mean + std:.2f}')
    plt.axvline(mean - std, color='green', linestyle='dashed', linewidth=2, label=f'-1 Std Dev: {mean - std:.2f}')

    # Add titles and labels
    plt.title('Distribution of Returns')
    plt.xlabel('Return (%)')
    plt.ylabel('Density')
    plt.legend()

    plt.show()
