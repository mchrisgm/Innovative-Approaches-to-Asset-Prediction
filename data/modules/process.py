
__all__ = ['process']


import os
import logging
from pathlib import Path

import pandas as pd


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


class DataComposer:
    equity_asset = Asset()
    currency_asset = Asset()
    bond_asset = Asset()

    def __init__(self, directory="./data/unprocessed",
                 save_directory="./data/processed",
                 filenames: dict[str, str] = {"equity": "",
                                              "currency": "",
                                              "bond": ""}):
        self.directory = directory
        self.save_directory = save_directory
        self.equity_filename = filenames.get("equity", "")
        self.currency_filename = filenames.get("currency", "")
        self.bond_filename = filenames.get("bond", "")

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
        Path(output_filename).mkdir(parents=True, exist_ok=True)


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
