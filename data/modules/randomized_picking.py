import os
import random
import pandas as pd
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from graph import create_image


RANDOM_ASSETS = 10
LOOKBACK = 5
IMAGE_SIZE = 64
CHANNELS = 3
MONOTONIC_OUTPUT = False


def get_random_dfs(path, number=10, specific=None) -> list[pd.DataFrame]:
    files = sorted(os.listdir(path))
    if not specific:
        random_files = random.sample(files, number)
    else:
        random_files = []
        for specific_file in sorted(specific):
            specific_file: str = specific_file.lower().strip()
            for file in files:
                file: str = file.lower().strip()
                if file.startswith(specific_file):
                    random_files.append(file)
                    break
    print("Random files:")
    print(*random_files, sep="\n")
    dfs: list[pd.DataFrame] = []
    for file in random_files:
        try:
            df: pd.DataFrame = pd.read_csv(f"{path}/{file}", parse_dates=["Date"])
            dfs.append((file, df))
        except Exception as e:
            print(f"Error reading {file}: {e}")
    return dfs


def outcomes(data: list[pd.DataFrame], monotonic: bool = False) -> list[pd.DataFrame]:
    outcome_dfs: list[pd.DataFrame] = []
    for filename, df in data:
        outcome_df = calculate_outcome(df, monotonic=monotonic)
        outcome_dfs.append((filename, outcome_df))
    return outcome_dfs


def calculate_outcome(df: pd.DataFrame, monotonic: bool = False) -> pd.DataFrame:
    # Calculate the percentage change in Close prices
    df["pct"] = df["Close"].pct_change()
    # Calculate the volume percentage
    df["Mov"] = (np.abs(((df["Close"] - df["Open"]) /
                        (df["High"] - df["Low"]))).round(2) if not monotonic else 1) * np.sign(df["pct"])
    # Replace infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Drop rows with NaN values in 'pct' and 'Mov' columns
    df.dropna(subset=["pct", "Mov"], inplace=True)
    return df


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
    for start in range(len(df) - lookback + 2):
        end = start + lookback + 1
        lookback_window = df.iloc[start:end].reset_index()  # Reset the index to have Date as a column  # noqa
        windows.append(lookback_window)
    return windows


def process_row(lookback_window: pd.DataFrame, lookback=5, image_size=64, channels=1) -> np.ndarray:
    image_df = lookback_window.head(lookback)
    outcome = lookback_window.tail(1)
    image = create_image(equity_df=image_df, bond_df=None, currency_df=None, width=image_size, height=image_size, lookback=lookback, rgb_channels=channels)  # noqa
    return {"image": image, "outcome": outcome["Mov"].values[0]}


def process_asset(filename, asset_windows, lookback=5, image_size=64, channels=1, desc="Processing windows") -> list[dict]:
    rows = []
    for window in tqdm(asset_windows, desc=desc, leave=False):
        row = process_row(window, lookback=lookback, image_size=image_size, channels=channels)
        rows.append(row)
    return filename, rows


def main(path="./data/unprocessed/US", save_dir="./data/processed", number=10, lookback=5, image_size=64, channels=1, monotonic=False, specific=None, name="RANDOM"):
    data = get_random_dfs(path=path, number=number, specific=specific)
    data = outcomes(data, monotonic=monotonic)
    assets = [(filename, lookback_windows(df, lookback=lookback)) for filename, df in data]

    rows = []
    with ThreadPoolExecutor(max_workers=len(assets)) as executor:
        futures = []
        for filename, asset_windows in assets:
            futures.append(executor.submit(process_asset, filename, asset_windows, lookback, image_size, channels, f"Processing {filename}"))

        for future in as_completed(futures):
            filename, result = future.result()
            rows.extend(result)
            tqdm.write(f"{filename} completed.", end="\n")

    final_df = pd.DataFrame(rows)

    output_filename = f"{save_dir}/US.{name}.{number}.{lookback}.{image_size}.{channels}.{'MONO' if monotonic else 'RANGE'}"   # noqa

    output_path = Path(output_filename)
    if output_path.exists() and output_path.is_dir():
        shutil.rmtree(output_path)
    Path(output_filename).mkdir(parents=True, exist_ok=True)

    np.save(f"{output_filename}/data.npy", final_df.to_numpy(), allow_pickle=True)


if __name__ == '__main__':
    main(path="./data/unprocessed/US", number=RANDOM_ASSETS,
         lookback=LOOKBACK, image_size=IMAGE_SIZE,
         channels=CHANNELS, monotonic=MONOTONIC_OUTPUT,
         specific=["aapl", "msft", "amzn",
                   "nvda", "googl", "tsla",
                   "goog", "brk-b", "fb",
                   "unh"],
         name="TOP10")
