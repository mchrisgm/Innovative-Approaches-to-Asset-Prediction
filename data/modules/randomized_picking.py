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
LOOKBACK = 4
IMAGE_SIZE = 64
CHANNELS = 3
MONOTONIC_OUTPUT = False
VOL_LOOKBACK = 10  # Constant for volatility calculation
STD_VARIATION = 1.0  # Constant for standard deviation threshold

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
    print("Selected files:")
    print(*random_files, sep="\n")
    dfs: list[pd.DataFrame] = []
    for file in random_files:
        try:
            df: pd.DataFrame = pd.read_csv(f"{path}/{file}", parse_dates=["Date"])
            dfs.append((file, df))
        except Exception as e:
            print(f"Error reading {file}: {e}")
    return dfs

def outcomes(data: list[pd.DataFrame], monotonic: bool = False, vol: bool = True) -> list[pd.DataFrame]:
    outcome_dfs: list[pd.DataFrame] = []
    for filename, df in data:
        if vol:
            outcome_df = calculate_vol_outcome(df, std_variation=STD_VARIATION, n_days=VOL_LOOKBACK)
        else:
            outcome_df = calculate_outcome(df, monotonic=monotonic)
        outcome_dfs.append((filename, outcome_df))
    return outcome_dfs

def calculate_outcome(df: pd.DataFrame, monotonic: bool = False) -> pd.DataFrame:
    df["pct"] = df["Close"].pct_change()
    df["Mov"] = (np.abs(((df["Close"] - df["Open"]) /
                        (df["High"] - df["Low"]))).round(2) if not monotonic else 1) * np.sign(df["pct"])
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=["pct", "Mov"], inplace=True)
    return df

def calculate_vol_outcome(df: pd.DataFrame, std_variation: float = 1.0, n_days: int = 20) -> pd.DataFrame:
    df['pct_change'] = df['Close'].pct_change()
    df['rolling_std'] = df['pct_change'].rolling(window=n_days).std()
    df['rolling_mean'] = df['pct_change'].rolling(window=n_days).mean()
    df['Mov'] = np.where(
        df['pct_change'] > df['rolling_mean'] + std_variation * df['rolling_std'], 1,
        np.where(
            df['pct_change'] < df['rolling_mean'] - std_variation * df['rolling_std'], 1,
            0
        )
    )
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=["pct_change", "Mov"], inplace=True)
    df.dropna(inplace=True)
    return df

def lookback_windows(df: pd.DataFrame, lookback=5, vol_lookback=20) -> list[pd.DataFrame]:
    """
    Generate a list of DataFrames, each representing a rolling lookback window.
    Ensures that each window has enough data for volatility calculations.
    """
    total_lookback = max(lookback, vol_lookback)
    windows = []
    for start in range(len(df) - total_lookback + 1):
        end = start + total_lookback + 1
        lookback_window = df.iloc[start:end].reset_index(drop=True)
        windows.append(lookback_window)
    return windows

def process_row(lookback_window: pd.DataFrame, lookback=5, image_size=64, channels=1) -> np.ndarray:
    image_df = lookback_window.tail(lookback+1).head(lookback)
    outcome = lookback_window.tail(1)
    image = create_image(equity_df=image_df, bond_df=None, currency_df=None, width=image_size, height=image_size, lookback=lookback, rgb_channels=channels)
    return {"image": image, "outcome": outcome["Mov"].values[0]}

def process_asset(filename, asset_windows, lookback=5, image_size=64, channels=1, desc="Processing windows") -> list[dict]:
    rows = []
    for window in tqdm(asset_windows, desc=desc, leave=False):
        row = process_row(window, lookback=lookback, image_size=image_size, channels=channels)
        rows.append(row)
    return filename, rows

def main(path="./data/unprocessed/US", save_dir="./data/processed", number=10, lookback=5, image_size=64, channels=1, monotonic=False, specific=None, name="RANDOM", vol=True):
    data = get_random_dfs(path=path, number=number, specific=specific)
    data = outcomes(data, monotonic=monotonic, vol=vol)
    
    min_required_rows = max(VOL_LOOKBACK, lookback) + 1
    assets = []
    for filename, df in data:
        if len(df) >= min_required_rows:
            assets.append((filename, lookback_windows(df, lookback=lookback, vol_lookback=VOL_LOOKBACK)))
        else:
            print(f"Skipping {filename} due to insufficient data (needed {min_required_rows}, got {len(df)})")

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

    output_filename = f"{save_dir}/US.{name}.{number}.{lookback}.{image_size}.{channels}.{'MONO' if monotonic else 'RANGE'}.{'VOL' if vol else 'REG'}"

    output_path = Path(output_filename)
    if output_path.exists() and output_path.is_dir():
        shutil.rmtree(output_path)
    Path(output_filename).mkdir(parents=True, exist_ok=True)

    np.save(f"{output_filename}/data.npy", final_df.to_numpy(), allow_pickle=True)

    print(f"Saved to {output_filename}")
    return final_df

if __name__ == '__main__':
    output = main(path="./data/unprocessed/US", 
         number=RANDOM_ASSETS,
         lookback=LOOKBACK, 
         image_size=IMAGE_SIZE,
         channels=CHANNELS, 
         monotonic=MONOTONIC_OUTPUT,
         specific=None,  # Set to None for random selection, or provide a list of specific stocks
         name="RANDOM",
         vol=True)

    print(output.head())
