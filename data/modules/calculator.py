import pandas as pd
import numpy as np


__all__ = ["outcomes"]


def outcomes(equity_df: pd.DataFrame, currency_df: pd.DataFrame, bond_df: pd.DataFrame) -> pd.DataFrame:    # noqa
    # Calculate the percentage change of each dataframe
    equity_df["pct"] = equity_df["Close"].pct_change(fill_method=None)
    currency_df["pct"] = currency_df["Close"].pct_change(fill_method=None)
    bond_df["pct"] = bond_df["Index"].pct_change(fill_method=None)

    # Select the Date and pct columns from each dataframe
    equity_pct = equity_df[["Date", "pct"]] \
        .rename(columns={"pct": "Equity"})
    currency_pct = currency_df[["Date", "pct"]] \
        .rename(columns={"pct": "Currency"})
    bond_pct = bond_df[["Date", "pct"]] \
        .rename(columns={"pct": "Bond"})

    # Combine the "pct" columns of the dataframes, matching the dates using an outer join   # noqa
    combined_df = pd.merge(equity_pct, currency_pct, on="Date", how="outer")
    combined_df = pd.merge(combined_df, bond_pct, on="Date", how="outer")

    # Set the Date column as the index
    combined_df.set_index("Date", inplace=True)

    # Drop the NaN values
    combined_df.dropna(inplace=True)

    # Get the cumulative sum of the positives and negatives
    combined_df["Long"] = combined_df[combined_df > 0].sum(axis=1)
    combined_df["Short"] = -combined_df[combined_df < 0].sum(axis=1)

    # Calculate the movements based on percentage changes and positions
    combined_df["Equity Movement"] = combined_df["Equity"] / np.where(combined_df["Equity"] > 0, combined_df["Long"], combined_df["Short"])         # noqa
    combined_df["Currency Movement"] = combined_df["Currency"] / np.where(combined_df["Currency"] > 0, combined_df["Long"], combined_df["Short"])   # noqa
    combined_df["Bond Movement"] = combined_df["Bond"] / np.where(combined_df["Bond"] > 0, combined_df["Long"], combined_df["Short"])               # noqa

    # Drop the columns that are not needed
    combined_df.drop(
        columns=["Equity", "Currency", "Bond", "Long", "Short"],
        inplace=True)

    return combined_df


if __name__ == "__main__":
    # Load the data
    equity_df = pd.read_csv(
        "data/unprocessed/EQUITY.SP500.2000.2023.csv",
        parse_dates=["Date"])
    currency_df = pd.read_csv(
        "data/unprocessed/CURRENCY.EURUSD.2004.2023.csv",
        parse_dates=["Date"])
    bond_df = pd.read_csv(
        "data/unprocessed/BOND.USTREASURYINDEX.2014.2024.csv",
        parse_dates=["Date"])

    print(outcomes(equity_df, currency_df, bond_df).head())