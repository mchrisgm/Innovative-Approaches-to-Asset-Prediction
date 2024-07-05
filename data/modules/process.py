__all__ = ['process']

import os
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from graph import create_image
from calculator import outcomes


class Asset:
    """
    A class to represent an asset with financial data.

    Attributes:
    name (str): Name of the asset.
    data (pd.DataFrame): DataFrame containing the asset data.

    Methods:
    set(data: pd.DataFrame) -> None: Set the asset data.
    get() -> pd.DataFrame: Get the asset data.
    get_start_year() -> pd.Timestamp: Get the earliest date in the asset data.
    get_end_year() -> pd.Timestamp: Get the latest date in the asset data.
    crop(start_year: str, end_year: str) -> None: Crop the asset data to the specified date range.
    """ # noqa
    def __init__(self, name: str = "ASSET"):
        self.name = name
        self.start_year = None
        self.end_year = None
        self.data: pd.DataFrame = pd.DataFrame()

    def set(self, data: pd.DataFrame) -> None:
        """
        Set the asset data.

        Parameters:
        data (pd.DataFrame): DataFrame containing the asset data.
        """
        self.data: pd.DataFrame = data

    def get(self) -> pd.DataFrame:
        """
        Get the asset data.

        Returns:
        pd.DataFrame: DataFrame containing the asset data.
        """
        return self.data

    def get_start_year(self) -> pd.Timestamp:
        """
        Get the earliest date in the asset data.

        Returns:
        pd.Timestamp: The earliest date in the asset data.
        """
        return self.data["Date"].min()

    def get_end_year(self) -> pd.Timestamp:
        """
        Get the latest date in the asset data.

        Returns:
        pd.Timestamp: The latest date in the asset data.
        """
        return self.data["Date"].max()

    def crop(self, start_year: str, end_year: str) -> None:
        """
        Crop the asset data to the specified date range.

        Parameters:
        start_year (str): The start year for cropping.
        end_year (str): The end year for cropping.
        """
        start_date = pd.to_datetime(start_year)
        end_date = pd.to_datetime(end_year)
        self.data = self.data[(self.data["Date"] >= start_date) & (self.data["Date"] <= end_date)]  # noqa

    def __str__(self) -> str:
        return f"{self.name} ({len(self.data)} rows)"


def lookback_windows(df: pd.DataFrame, lookback=5) -> list[pd.DataFrame]:
    """
    Generate a list of DataFrames, each representing a rolling lookback window.

    Parameters:
    df (pd.DataFrame): DataFrame containing data with a Date column.
    lookback (int): The number of rows to include in each lookback window.

    Returns:
    list[pd.DataFrame]: List of DataFrames, each representing a lookback window with a Date column.
    """ # noqa
    df.set_index("Date", inplace=True)
    windows = []
    for start in range(len(df) - lookback + 1):
        end = start + lookback
        lookback_window = df.iloc[start:end].reset_index()  # Reset the index to have Date as a column  # noqa
        windows.append(lookback_window)
    return windows


class DataComposer:
    """
    A class to compose and process financial data for multiple assets.

    Attributes:
    equity_asset (Asset): The equity asset.
    currency_asset (Asset): The currency asset.
    bond_asset (Asset): The bond asset.
    lookback (int): The lookback period for creating rolling windows.
    image_size (int): The size of the generated images.
    n_images (int): The number of images to generate.
    save_png (bool): Whether to save the generated images as PNG files.

    Methods:
    file_exists(directory: str, filename: str): Check if a file exists.
    load(): Load the asset data from CSV files.
    save(): Process the data and save the results.
    """
    equity_asset = Asset()
    currency_asset = Asset()
    bond_asset = Asset()
    lookback = 5
    image_size = 64
    n_images = 0
    save_png = False

    def __init__(self, directory="./data/unprocessed",
                 save_directory="./data/processed",
                 lookback=5,
                 filenames: dict[str, str] = {"equity": "",
                                              "currency": "",
                                              "bond": ""}):
        """
        Initialize the DataComposer with directories, lookback period, and filenames.

        Parameters:
        directory (str): The directory containing the unprocessed data files.
        save_directory (str): The directory to save the processed data files.
        lookback (int): The lookback period for creating rolling windows.
        filenames (dict[str, str]): Dictionary containing the filenames for equity, currency, and bond data.
        """ # noqa
        self.directory = directory
        self.save_directory = save_directory
        self.equity_filename = filenames.get("equity", "")
        self.currency_filename = filenames.get("currency", "")
        self.bond_filename = filenames.get("bond", "")
        self.lookback = self.lookback if lookback == 5 else lookback

        if not self.file_exists(directory, self.equity_filename):
            logging.error(f"File not found: {self.equity_filename}")
            raise FileNotFoundError(f"File not found: {self.equity_filename}")
        if not self.file_exists(directory, self.currency_filename):
            logging.error(f"File not found: {self.currency_filename}")
            raise FileNotFoundError(f"File not found: {self.currency_filename}")
        if not self.file_exists(directory, self.bond_filename):
            logging.error(f"File not found: {self.bond_filename}")
            raise FileNotFoundError(f"File not found: {self.bond_filename}")

        equity_name = self.equity_filename.split(".")[1]
        currency_name = self.currency_filename.split(".")[1]
        bond_name = self.bond_filename.split(".")[1]

        self.equity_asset = Asset(name=equity_name)
        self.currency_asset = Asset(name=currency_name)
        self.bond_asset = Asset(name=bond_name)

    def file_exists(self, directory: str, filename: str):
        """
        Check if a file exists in the specified directory.

        Parameters:
        directory (str): The directory to check.
        filename (str): The filename to check.

        Returns:
        bool: True if the file exists, False otherwise.
        """
        return os.path.isfile(os.path.join(os.getcwd(), directory, filename))

    def load(self):
        """
        Load the asset data from CSV files and crop to the effective date range.
        """ # noqa
        equity_data = pd.read_csv(os.path.join(self.directory, self.equity_filename), parse_dates=["Date"])         # noqa
        currency_data = pd.read_csv(os.path.join(self.directory, self.currency_filename), parse_dates=["Date"])     # noqa
        bond_data = pd.read_csv(os.path.join(self.directory, self.bond_filename), parse_dates=["Date"])             # noqa

        self.equity_asset.set(equity_data)
        self.currency_asset.set(currency_data)
        self.bond_asset.set(bond_data)

        self.effective_start_year = max(self.equity_asset.get_start_year(), self.currency_asset.get_start_year(), self.bond_asset.get_start_year())     # noqa
        self.effective_end_year = min(self.equity_asset.get_end_year(), self.currency_asset.get_end_year(), self.bond_asset.get_end_year())             # noqa

        self.equity_asset.crop(self.effective_start_year, self.effective_end_year)      # noqa
        self.currency_asset.crop(self.effective_start_year, self.effective_end_year)    # noqa
        self.bond_asset.crop(self.effective_start_year, self.effective_end_year)        # noqa

    def save(self):
        """
        Process the data and save the results, including images and movement data.
        """ # noqa
        output_filename = f"{self.save_directory}/{self.equity_asset.name}.{self.currency_asset.name}.{self.bond_asset.name}.{self.effective_start_year.year}.{self.effective_end_year.year}"   # noqa
        output_path = Path(output_filename)
        if output_path.exists() and output_path.is_dir():
            shutil.rmtree(output_path)
        Path(output_filename).mkdir(parents=True, exist_ok=True)

        outcomes_df = outcomes(self.equity_asset.get(),
                               self.currency_asset.get(),
                               self.bond_asset.get())

        # Remove the first self.lookback rows of the outcomes_df
        outcomes_df = outcomes_df.iloc[self.lookback-1:]

        if self.n_images > 0:
            outcomes_df = outcomes_df.sample(n=self.n_images)

        equity_lookback = lookback_windows(self.equity_asset.get(),
                                           self.lookback)
        currency_lookback = lookback_windows(self.currency_asset.get(),
                                             self.lookback)
        bond_lookback = lookback_windows(self.bond_asset.get(),
                                         self.lookback)

        final_df = pd.DataFrame({
            "Image": pd.Series(dtype='object'),
            "Equity": pd.Series(dtype='float'),
            "Currency": pd.Series(dtype='float'),
            "Bond": pd.Series(dtype='float')
        })
        progress = 1
        for equity, currency, bond, outcome in zip(equity_lookback, currency_lookback, bond_lookback, outcomes_df.iterrows()):      # noqa
            print(f"{progress / len(outcomes_df) * 100:.2f}%  -> {progress} images created")                                        # noqa
            outcome_image = create_image(equity, currency, bond, width=self.image_size, height=self.image_size)                     # noqa
            new_row = {
                "Image": outcome_image,
                "Equity": outcome[1]["Equity Movement"],
                "Currency": outcome[1]["Currency Movement"],
                "Bond": outcome[1]["Bond Movement"]
            }
            final_df = pd.concat([final_df, pd.DataFrame([new_row])], ignore_index=True)    # noqa
            if self.save_png:
                Image.fromarray(outcome_image).save(f"{output_filename}/{progress}.png")    # noqa
            progress += 1

        final_df.to_numpy().dump(f"{output_filename}/data.npy")


def process(unprocessed_folder: str = "./data/unprocessed",
            equity: str = "SP500",
            currency: str = "EURUSD",
            bond: str = "USTREASURYINDEX",
            start_year: int = 2000,
            end_year: int = 2024):
    """
    Process the unprocessed data files and generate the necessary data.

    Parameters:
    unprocessed_folder (str): Directory containing the unprocessed data files.
    equity (str): Name of the equity data file.
    currency (str): Name of the currency data file.
    bond (str): Name of the bond data file.
    start_year (int): The start year for processing data.
    end_year (int): The end year for processing data.
    """
    equities: list[str] = []
    currencies: list[str] = []
    bonds: list[str] = []
    for filename in os.listdir(unprocessed_folder):
        if filename.endswith(".csv"):
            if filename.lower().startswith("equity"):
                equities.append(filename)
            elif filename.lower().startswith("currency"):
                currencies.append(filename)
            elif filename.lower().startswith("bond"):
                bonds.append(filename)
            else:
                print(f"Unknown file type: {filename}")

    equities = [e for e in equities if equity in e]
    currencies = [c for c in currencies if currency in c]
    bonds = [b for b in bonds if bond in b]

    equities = [e for e in equities if start_year <= int(e.split(".")[2]) and int(e.split(".")[3]) <= end_year] or [""]     # noqa
    currencies = [c for c in currencies if start_year <= int(c.split(".")[2]) and int(c.split(".")[3]) <= end_year] or [""] # noqa
    bonds = [b for b in bonds if start_year <= int(b.split(".")[2]) and int(b.split(".")[3]) <= end_year] or [""]           # noqa

    equity = max(equities)
    currency = max(currencies)
    bond = max(bonds)

    names: dict[str, str] = {"equity": equity,
                             "currency": currency,
                             "bond": bond}

    composer = DataComposer(directory=unprocessed_folder,
                            filenames=names)
    composer.load()
    composer.save()


if __name__ == "__main__":
    process()
    data = np.load("./data/processed/SP500.EURUSD.USTREASURYINDEX.2014.2023/data.npy", allow_pickle=True)   # noqa
    print(data)
