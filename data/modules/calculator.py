import pandas as pd
import numpy as np

__all__ = ["outcomes"]

def outcomes(equity_df: pd.DataFrame = None, currency_df: pd.DataFrame = None, bond_df: pd.DataFrame = None, monotonic: bool = False) -> pd.DataFrame:
    """
    Calculate the percentage change and movements for equity, currency, and bond data.

    Parameters:
    equity_df (pd.DataFrame): DataFrame containing equity data with Date and Close columns.
    currency_df (pd.DataFrame): DataFrame containing currency data with Date and Close columns.
    bond_df (pd.DataFrame): DataFrame containing bond index data with Date and Index columns.

    Returns:
    pd.DataFrame: Combined DataFrame with movements calculated for equity, currency, and bond data.
    """
    # Initialize an empty DataFrame to hold the combined percentage changes
    combined_df = pd.DataFrame(columns=["Date"])

    if equity_df is not None and not equity_df.empty:
        equity_df["pct"] = equity_df["Close"].pct_change(fill_method=None)
        equity_pct = equity_df[["Date", "pct"]].rename(columns={"pct": "Equity"})
        combined_df = equity_pct if combined_df.empty else pd.merge(combined_df, equity_pct, on="Date", how="outer")

    if currency_df is not None and not currency_df.empty:
        currency_df["pct"] = currency_df["Close"].pct_change(fill_method=None)
        currency_pct = currency_df[["Date", "pct"]].rename(columns={"pct": "Currency"})
        combined_df = currency_pct if combined_df.empty else pd.merge(combined_df, currency_pct, on="Date", how="outer")

    if bond_df is not None and not bond_df.empty:
        bond_df["pct"] = bond_df["Index"].pct_change(fill_method=None)
        bond_pct = bond_df[["Date", "pct"]].rename(columns={"pct": "Bond"})
        combined_df = bond_pct if combined_df.empty else pd.merge(combined_df, bond_pct, on="Date", how="outer")

    # Set the Date column as the index
    combined_df.set_index("Date", inplace=True)

    # Drop the NaN values
    combined_df.dropna(inplace=True)

    if monotonic:
        # Calculate the movements based on percentage changes
        if equity_df is not None and not equity_df.empty:
            combined_df["Equity Movement"] = np.where(combined_df.get("Equity", 0) > 0, 1, -1)
        if currency_df is not None and not currency_df.empty:
            combined_df["Currency Movement"] = np.where(combined_df.get("Currency", 0) > 0, 1, -1)
        if bond_df is not None and not bond_df.empty:
            combined_df["Bond Movement"] = np.where(combined_df.get("Bond", 0) > 0, 1, -1)
    else:
        # Calculate the cumulative sum of positive and negative values for each date            # noqa
        combined_df["Long"] = combined_df[combined_df > 0].sum(axis=1)
        combined_df["Short"] = -combined_df[combined_df < 0].sum(axis=1)

        # Calculate the movements based on percentage changes and positions
        if equity_df is not None and not equity_df.empty:
            combined_df["Equity Movement"] = combined_df["Equity"] / np.where(combined_df["Equity"] > 0, combined_df["Long"], combined_df["Short"])         # noqa
        if currency_df is not None and not currency_df.empty:
            combined_df["Currency Movement"] = combined_df["Currency"] / np.where(combined_df["Currency"] > 0, combined_df["Long"], combined_df["Short"])
        if bond_df is not None and not bond_df.empty:
            combined_df["Bond Movement"] = combined_df["Bond"] / np.where(combined_df["Bond"] > 0, combined_df["Long"], combined_df["Short"])

    # Drop the columns that are not needed
    combined_df.drop(columns=["Equity", "Currency", "Bond", "Long", "Short"], inplace=True, errors='ignore')

    print(combined_df.tail(5))
    return combined_df

if __name__ == "__main__":
    # Load the data from CSV files
    equity_df = pd.read_csv("data/unprocessed/EQUITY.SP500.2000.2023.csv",
                            parse_dates=["Date"])
    currency_df = pd.read_csv("data/unprocessed/CURRENCY.EURUSD.2004.2023.csv",
                              parse_dates=["Date"])
    bond_df = pd.read_csv("data/unprocessed/BOND.USTREASURYINDEX.2014.2024.csv",
                          parse_dates=["Date"])

    # Print the first few rows of the outcomes DataFrame
    print(outcomes(equity_df, currency_df, bond_df).head())
