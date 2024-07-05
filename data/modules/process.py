
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
    def __init__(self, name: str = "ASSET"):
        self.name = name
        self.start_year = None
        self.end_year = None
        self.data: pd.DataFrame = pd.DataFrame()

    def set(self, data: pd.DataFrame) -> None:
        self.data: pd.DataFrame = data

    def get(self) -> pd.DataFrame:
        return self.data

    def get_start_year(self) -> pd.Timestamp:
        return self.data["Date"].min()

    def get_end_year(self) -> pd.Timestamp:
        return self.data["Date"].max()

    def crop(self, start_year: str, end_year: str) -> None:
        # Convert start_year and end_year to datetime objects
        start_date = pd.to_datetime(start_year)
        end_date = pd.to_datetime(end_year)

        # Filter the data
        self.data = self.data[(self.data["Date"] >= start_date) &
                              (self.data["Date"] <= end_date)]

    def __str__(self) -> str:
        return f"{self.name} ({len(self.data)} rows)"


def lookback_windows(df: pd.DataFrame, lookback=5) -> list[pd.DataFrame]:
    # Ensure the Date column is the index
    df.set_index("Date", inplace=True)

    # Create an empty list to store the lookback windows
    windows = []

    # Iterate through each possible start index for the rolling windows
    for start in range(len(df) - lookback + 1):
        end = start + lookback
        lookback_window = df.iloc[start:end].reset_index()  # Reset the index to have Date as a column
        windows.append(lookback_window)
    return windows


class DataComposer:
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
        return os.path.isfile(os.path.join(
                                  os.getcwd(),
                                  directory,
                                  filename))

    def load(self):
        equity_data = pd.read_csv(os.path.join(self.directory,
                                               self.equity_filename),
                                  parse_dates=["Date"])
        currency_data = pd.read_csv(os.path.join(self.directory,
                                                 self.currency_filename),
                                    parse_dates=["Date"])
        bond_data = pd.read_csv(os.path.join(self.directory,
                                             self.bond_filename),
                                parse_dates=["Date"])

        self.equity_asset.set(equity_data)
        self.currency_asset.set(currency_data)
        self.bond_asset.set(bond_data)

        self.effective_start_year = max(self.equity_asset.get_start_year(),
                                        self.currency_asset.get_start_year(),
                                        self.bond_asset.get_start_year())
        self.effective_end_year = min(self.equity_asset.get_end_year(),
                                      self.currency_asset.get_end_year(),
                                      self.bond_asset.get_end_year())

        self.equity_asset.crop(self.effective_start_year,
                               self.effective_end_year)
        self.currency_asset.crop(self.effective_start_year,
                                 self.effective_end_year)
        self.bond_asset.crop(self.effective_start_year,
                             self.effective_end_year)

    def save(self):
        output_filename = f"{self.save_directory}/{self.equity_asset.name}.{self.currency_asset.name}.{self.bond_asset.name}.{self.effective_start_year.year}.{self.effective_end_year.year}"
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

        # numpy array = create_image(equity_df_with_lookback, currency_df_with_lookback, bond_df_with_lookback)
        equity_lookback = lookback_windows(self.equity_asset.get(), self.lookback)
        currency_lookback = lookback_windows(self.currency_asset.get(), self.lookback)
        bond_lookback = lookback_windows(self.bond_asset.get(), self.lookback)

        final_df = pd.DataFrame({
            "Image": pd.Series(dtype='object'),
            "Equity": pd.Series(dtype='float'),
            "Currency": pd.Series(dtype='float'),
            "Bond": pd.Series(dtype='float')
        })
        progress = 1
        for equity, currency, bond, outcome in zip(equity_lookback, currency_lookback, bond_lookback, outcomes_df.iterrows()):
            print(f"{progress / len(outcomes_df) * 100:.2f}%  -> {progress} images created")
            outcome_image = create_image(equity, currency, bond, width=self.image_size, height=self.image_size)
            new_row = {
                "Image": outcome_image,
                "Equity": outcome[1]["Equity Movement"],
                "Currency": outcome[1]["Currency Movement"],
                "Bond": outcome[1]["Bond Movement"]
            }
            final_df = pd.concat([final_df, pd.DataFrame([new_row])], ignore_index=True)
            if self.save_png:
                Image.fromarray(outcome_image).save(f"{output_filename}/{progress}.png")
            progress += 1
        # final_df.to_pickle(f"{output_filename}/data.pkl")
        final_df.to_numpy().dump(f"{output_filename}/data.npy")


def process(unprocessed_folder: str = "./data/unprocessed",
            equity: str = "SP500",
            currency: str = "EURUSD",
            bond: str = "USTREASURYINDEX",
            start_year: int = 2000,
            end_year: int = 2024):
    # Read all files in the "unprocessed" folder
    equities: list[str] = []
    currencies: list[str] = []
    bonds: list[str] = []
    for filename in os.listdir(unprocessed_folder):
        if filename.endswith(".csv"):
            # Process the file
            if filename.lower().startswith("equity"):
                equities.append(filename)
            elif filename.lower().startswith("currency"):
                currencies.append(filename)
            elif filename.lower().startswith("bond"):
                bonds.append(filename)
            else:
                print(f"Unknown file type: {filename}")

    # Find the files that match the given names
    equities = [e for e in equities if equity in e]
    currencies = [c for c in currencies if currency in c]
    bonds = [b for b in bonds if bond in b]

    # Filter the files by year
    equities = [e for e in equities if start_year <= int(e.split(".")[2])
                and int(e.split(".")[3]) <= end_year] or [""]
    currencies = [c for c in currencies if start_year <= int(c.split(".")[2])
                  and int(c.split(".")[3]) <= end_year] or [""]
    bonds = [b for b in bonds if start_year <= int(b.split(".")[2])
             and int(b.split(".")[3]) <= end_year] or [""]

    # Get the latest files
    equity = max(equities)
    currency = max(currencies)
    bond = max(bonds)

    names: dict[str, str] = {"equity": equity,
                             "currency": currency,
                             "bond": bond}

    # Compose the data
    composer = DataComposer(directory=unprocessed_folder,
                            filenames=names)
    composer.load()
    composer.save()


if __name__ == "__main__":
    process()
    data = np.load("./data/processed/SP500.EURUSD.USTREASURYINDEX.2014.2023/data.npy", allow_pickle=True)
    print(data)
