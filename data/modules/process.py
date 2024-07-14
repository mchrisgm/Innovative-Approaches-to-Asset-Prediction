__all__ = ['process']

import os
import logging
import shutil
from pathlib import Path
from tqdm import tqdm
import time
import warnings

import numpy as np
import pandas as pd
from PIL import Image

from graph import create_image
from calculator import outcomes

warnings.simplefilter(action='ignore', category=Warning)

IMAGE_SIZE = 64
LOOKBACK_PERIOD = 9
SAVE_PGNS = False
RGB_CHANNELS = 3
MONOTONIC = False
NO_IMAGES_TO_GENERATE = 0


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

    def crop(self, start_year: str, end_year: str, date_index=None) -> None:
        """
        Crop the asset data to the specified date range.

        Parameters:
        start_year (str): The start year for cropping.
        end_year (str): The end year for cropping.
        """
        start_date = pd.to_datetime(start_year)
        end_date = pd.to_datetime(end_year)
        self.data = self.data[(self.data["Date"] >= start_date) & (self.data["Date"] <= end_date)]  # noqa

        # Only keep the dates that are in the date_index
        if date_index is not None:
            self.data = self.data[self.data["Date"].isin(date_index)]
            #Fill the dates missing from self.data but exist in date_index
            missing_dates = date_index[~date_index.isin(self.data["Date"])]
            missing_data = pd.DataFrame({"Date": missing_dates})
            self.data = pd.concat([self.data, missing_data], ignore_index=True)
            self.data = self.data.sort_values("Date")

        self.data = self.data.set_index("Date").reset_index()

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
    lookback = LOOKBACK_PERIOD
    image_size = IMAGE_SIZE
    n_images = NO_IMAGES_TO_GENERATE
    save_png = SAVE_PGNS
    rgb_channels = RGB_CHANNELS
    monotonic = MONOTONIC

    def __init__(self, directory="./data/unprocessed",
                 save_directory="./data/processed",
                 lookback=5,
                 filenames: dict[str, str] = {"equity": None,
                                              "currency": None,
                                              "bond": None}):
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
        self.equity_filename = filenames.get("equity", None)
        self.currency_filename = filenames.get("currency", None)
        self.bond_filename = filenames.get("bond", None)
        self.lookback = self.lookback if lookback == 5 else lookback

        if self.equity_filename and not self.file_exists(directory, self.equity_filename):
            logging.error(f"File not found: {self.equity_filename}")
            raise FileNotFoundError(f"File not found: {self.equity_filename}")
        if self.currency_filename and not self.file_exists(directory, self.currency_filename):
            logging.error(f"File not found: {self.currency_filename}")
            raise FileNotFoundError(f"File not found: {self.currency_filename}")
        if self.bond_filename and not self.file_exists(directory, self.bond_filename):
            logging.error(f"File not found: {self.bond_filename}")
            raise FileNotFoundError(f"File not found: {self.bond_filename}")

        if self.equity_filename:
            equity_name = self.equity_filename.split(".")[1]
            self.equity_asset = Asset(name=equity_name)

        if self.currency_filename:
            currency_name = self.currency_filename.split(".")[1]
            self.currency_asset = Asset(name=currency_name)
        
        if self.bond_filename:
            bond_name = self.bond_filename.split(".")[1]
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

        def load_equity():
            equity_data = pd.read_csv(os.path.join(self.directory, self.equity_filename), parse_dates=["Date"])         # noqa
            self.equity_asset.set(equity_data)
        
        def load_currency():
            currency_data = pd.read_csv(os.path.join(self.directory, self.currency_filename), parse_dates=["Date"])
            self.currency_asset.set(currency_data)

        def load_bond():
            bond_data = pd.read_csv(os.path.join(self.directory, self.bond_filename), parse_dates=["Date"])
            self.bond_asset.set(bond_data)

        if self.equity_filename:
            load_equity()
        if self.currency_filename:
            load_currency()
        if self.bond_filename:
            load_bond()

        self.effective_start_year = max(
            self.equity_asset.get_start_year() if self.equity_filename else pd.Timestamp.min,
            self.currency_asset.get_start_year() if self.currency_filename else pd.Timestamp.min,
            self.bond_asset.get_start_year() if self.bond_filename else pd.Timestamp.min
        )

        self.effective_end_year = min(
            self.equity_asset.get_end_year() if self.equity_filename else pd.Timestamp.max,
            self.currency_asset.get_end_year() if self.currency_filename else pd.Timestamp.max,
            self.bond_asset.get_end_year() if self.bond_filename else pd.Timestamp.max
        )

        if self.equity_filename:
            self.equity_asset.crop(self.effective_start_year, self.effective_end_year)
        if self.currency_filename:
            self.currency_asset.crop(self.effective_start_year, self.effective_end_year, self.equity_asset.get()["Date"] if self.equity_filename else None)
        if self.bond_filename:
            self.bond_asset.crop(self.effective_start_year, self.effective_end_year, self.equity_asset.get()["Date"] if self.equity_filename else None)

    def save(self):
        """
        Process the data and save the results, including images and movement data.
        """ # noqa
        output_filename = f"{self.save_directory}/{self.equity_asset.name}.{self.currency_asset.name}.{self.bond_asset.name}.{self.effective_start_year.year}.{self.effective_end_year.year}"   # noqa

        outcomes_df = outcomes(self.equity_asset.get() if self.equity_filename else pd.DataFrame(),
                               self.currency_asset.get() if self.currency_filename else pd.DataFrame(),
                               self.bond_asset.get() if self.bond_filename else pd.DataFrame(),
                               monotonic=self.monotonic)

        # Remove the first self.lookback rows of the outcomes_df
        outcomes_df = outcomes_df.iloc[self.lookback-1:]

        if self.n_images > 0:
            outcomes_df = outcomes_df.sample(n=self.n_images)

        equity_lookback = lookback_windows(self.equity_asset.get(), self.lookback) if self.equity_filename else []
        currency_lookback = lookback_windows(self.currency_asset.get(), self.lookback) if self.currency_filename else []
        bond_lookback = lookback_windows(self.bond_asset.get(), self.lookback) if self.bond_filename else []

        final_df = pd.DataFrame({
            "Image": pd.Series(dtype='object'),
            "Equity": pd.Series(dtype='float'),
            "Currency": pd.Series(dtype='float'),
            "Bond": pd.Series(dtype='float')
        })

        def process_row(equity: pd.DataFrame,
                        currency: pd.DataFrame,
                        bond: pd.DataFrame,
                        outcome: pd.DataFrame,
                        idx, save_png, image_size,
                        output_filename):
            outcome_image = create_image(equity, currency, bond, width=image_size, height=image_size, rgb_channels=self.rgb_channels)
            new_row = {
                "Image": outcome_image,
            }
            if outcome.get("Equity Movement", None) != None:
                new_row["Equity"] = outcome.get("Equity Movement")
            if outcome.get("Currency Movement", None) != None:
                new_row["Currency"] = outcome.get("Currency Movement")
            if outcome.get("Bond Movement", None) != None:
                new_row["Bond"] = outcome.get("Bond Movement")

            if save_png:
                Image.fromarray(outcome_image).save(f"{output_filename}/{idx}.png")
            return new_row

        args = [(equity, currency, bond, outcome[1], idx, self.save_png, self.image_size, output_filename) 
                for idx, (equity, currency, bond, outcome) in enumerate(zip(
                    equity_lookback if self.equity_filename else [None]*len(outcomes_df),
                    currency_lookback if self.currency_filename else [None]*len(outcomes_df),
                    bond_lookback if self.bond_filename else [None]*len(outcomes_df),
                    outcomes_df.iterrows()), 1)]

        rows = []
        
        start_time = time.time()
        etas = []
        for arg in tqdm(args, desc="Processing rows", unit="row"):
            row = process_row(*arg)
            rows.append(row)
            elapsed_time = time.time() - start_time
            rows_processed = len(rows)
            rows_remaining = len(args) - rows_processed
            time_per_row = elapsed_time / rows_processed
            try:
                etas.pop(0)
            except IndexError:
                pass
            if len(etas) <= 30 * int(rows_processed/elapsed_time):
                etas.append(rows_remaining * time_per_row)
            eta = sum(etas) / len(etas)

        final_df = pd.DataFrame(rows)

        output_path = Path(output_filename)
        if output_path.exists() and output_path.is_dir():
            shutil.rmtree(output_path)
        Path(output_filename).mkdir(parents=True, exist_ok=True)

        np.save(f"{output_filename}/data.npy", final_df.to_numpy(), allow_pickle=True)


def process(unprocessed_folder: str = "./data/unprocessed",
            equity: str = None,
            currency: str = None,
            bond: str = None,
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

    selected_equity = None
    selected_currency = None
    selected_bond = None

    if equity:
        equity_files = [e for e in equities if equity in e]
        equity_files = [e for e in equity_files if start_year <= int(e.split(".")[2]) <= end_year and start_year <= int(e.split(".")[3]) <= end_year]
        if equity_files:
            selected_equity = max(equity_files)

    if currency:
        currency_files = [c for c in currencies if currency in c]
        currency_files = [c for c in currency_files if start_year <= int(c.split(".")[2]) <= end_year and start_year <= int(c.split(".")[3]) <= end_year]
        if currency_files:
            selected_currency = max(currency_files)

    if bond:
        bond_files = [b for b in bonds if bond in b]
        bond_files = [b for b in bond_files if start_year <= int(b.split(".")[2]) <= end_year and start_year <= int(b.split(".")[3]) <= end_year]
        if bond_files:
            selected_bond = max(bond_files)

    print("Selected files:")
    print("Equity:  \t", selected_equity)
    print("Currency:\t", selected_currency)
    print("Bond:    \t", selected_bond)

    names: dict[str, str] = {"equity": selected_equity,
                             "currency": selected_currency,
                             "bond": selected_bond}

    composer = DataComposer(directory=unprocessed_folder,
                            filenames=names)
    composer.load()
    composer.save()


if __name__ == "__main__":
    equity = "SP500"
    currency = None
    bond = None
    start=2000
    end=2023
    process(equity=equity, currency=currency, bond=bond,
            start_year=start, end_year=end)
    data = np.load(f"./data/processed/{equity if equity else 'ASSET'}.{currency if currency else 'ASSET'}.{bond if bond else 'ASSET'}.{start}.{end}/data.npy", allow_pickle=True)   # noqa
    print(data[0][0].shape)
    Image.fromarray(data[0][0]).resize((512, 512)).show()
