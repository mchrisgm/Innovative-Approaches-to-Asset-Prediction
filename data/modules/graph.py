from io import BytesIO
import matplotlib.axes
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta

# Exported functions
__all__ = ["create_image"]

def create_candlestick(ax: matplotlib.axes.Axes, df):
    """
    Creates a candlestick chart on the given axis for the DataFrame.

    Parameters:
    ax (matplotlib.axes.Axes): The axis on which to plot the candlestick chart.
    df (pd.DataFrame): DataFrame containing the stock data with columns Date, Open, High, Low, Close.
    """
    assert "Date" in df.columns, "Date column not found in the DataFrame"
    assert "Open" in df.columns, "Open column not found in the DataFrame"
    assert "High" in df.columns, "High column not found in the DataFrame"
    assert "Low" in df.columns, "Low column not found in the DataFrame"
    assert "Close" in df.columns, "Close column not found in the DataFrame"
    for idx, row in df.iterrows():
        # high/low lines
        ax.plot([row['Date'], row['Date']], 
                [row['Low'], row['High']], 
                color='#0000FF', linewidth=0.5)
        # open marker
        ax.plot([row['Date']-timedelta(hours=3), row['Date']-timedelta(hours=10)], 
                [row['Open'], row['Open']], 
                color='#0000FF', linewidth=1)
        # close marker
        ax.plot([row['Date']+timedelta(hours=3), row['Date']+timedelta(hours=10)], 
                [row['Close'], row['Close']], 
                color='#0000FF', linewidth=1)

def create_line(ax, df):
    """
    Creates a line chart on the given axis for the DataFrame.

    Parameters:
    ax (matplotlib.axes.Axes): The axis on which to plot the line chart.
    df (pd.DataFrame): DataFrame containing the bond index data with columns Date, Index.
    """
    assert "Date" in df.columns, "Date column not found in the DataFrame"
    assert "Index" in df.columns, "Index column not found in the DataFrame"
    
    ax.plot(df['Date'], df['Index'], color='#0000FF', linewidth=1)

def make_figure(equity_df, currency_df, bond_df, lookback=5):
    """
    Creates a matplotlib figure with subplots for equity, currency, and bond data.

    Parameters:
    equity_df (pd.DataFrame): DataFrame containing the equity data.
    currency_df (pd.DataFrame): DataFrame containing the currency data.
    bond_df (pd.DataFrame): DataFrame containing the bond index data.

    Returns:
    matplotlib.figure.Figure: Matplotlib Figure object.
    """
    equity_df['Date'] = pd.to_datetime(equity_df['Date'], format='%d%b%Y:%H:%M:%S.%f')
    currency_df['Date'] = pd.to_datetime(currency_df['Date'], format='%d%b%Y:%H:%M:%S.%f')
    bond_df['Date'] = pd.to_datetime(bond_df['Date'], format='%d%b%Y:%H:%M:%S.%f')
    
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(0.3*lookback, 3.2), sharex=True)
    fig.patch.set_facecolor('black')
    for ax in axs:
        ax.set_facecolor('black')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d%b%Y:%H:%M:%S.%f'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

    create_candlestick(axs[0], equity_df)
    create_candlestick(axs[1], currency_df)
    create_line(axs[2], bond_df)

    plt.tight_layout()
    return fig

def create_image(equity_df, currency_df, bond_df, width=128, height=128, lookback=5):
    """
    Creates an image from the given dataframes and converts it to a grayscale numpy array.

    Parameters:
    equity_df (pd.DataFrame): DataFrame containing the equity data.
    currency_df (pd.DataFrame): DataFrame containing the currency data.
    bond_df (pd.DataFrame): DataFrame containing the bond index data.
    width (int): Width of the output image.
    height (int): Height of the output image.

    Returns:
    np.ndarray: Grayscale numpy array of the image.
    """
    # Create the matplotlib figure
    fig = make_figure(equity_df, currency_df, bond_df, lookback=lookback)

    # Save the figure to a BytesIO object as a PNG image
    img_bytes = BytesIO()
    fig.savefig(img_bytes, format="png", bbox_inches='tight', pad_inches=0, dpi=44.3)
    plt.close(fig)
    img_bytes.seek(0)

    # Open the image using PIL
    image = Image.open(img_bytes) \
                    # .resize((width, height))

    # Set all pixels to white (255) if the pixel value is greater than 20
    image = image.point(lambda p: p > 1 and 255)

    # Convert the image to a numpy array
    image_array = np.array(image)

    # Custom conversion to grayscale by applying weights to RGB channels
    grayscale_image_array = np.dot(image_array[..., :3], [0.3, 1, 1])
    grayscale_image_array = grayscale_image_array.astype(np.uint8)

    return grayscale_image_array

if __name__ == "__main__":
    # Create sample randomized data
    np.random.seed(0)
    date_period = 5
    dates = pd.date_range(start="2020-01-01", periods=date_period)

    # Create equity dataframe
    equity_df = pd.DataFrame({
        "Date": dates,
        "Open": np.random.randint(100, 200, date_period),
        "High": np.random.randint(200, 300, date_period),
        "Low": np.random.randint(50, 100, date_period),
        "Close": np.random.randint(100, 200, date_period)
    })

    # Create currency dataframe
    currency_df = pd.DataFrame({
        "Date": dates,
        "Open": np.random.randint(100, 200, date_period),
        "High": np.random.randint(200, 300, date_period),
        "Low": np.random.randint(50, 100, date_period),
        "Close": np.random.randint(100, 200, date_period)
    })

    # Create bond dataframe
    bond_df = pd.DataFrame({
        "Date": dates,
        "Index": np.random.randint(100, 200, date_period)
    })

    # Create grayscale image from the dataframes
    img = create_image(equity_df, currency_df, bond_df)
    print(img.shape)

    # Display and save the image
    pil_image = Image.fromarray(img)
    pil_image.show()
    pil_image.save("./data/sample_image_generation.png")
