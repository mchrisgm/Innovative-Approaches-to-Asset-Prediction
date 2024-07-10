from io import BytesIO
import numpy as np
import pandas as pd
from PIL import Image
from PIL.Image import Resampling
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Exported functions
__all__ = ["create_image"]


def create_candlestick(df) -> go.Ohlc:
    """
    Creates a candlestick chart for the given DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing the stock data with columns Date, Open, High, Low, Close.

    Returns:
    go.Ohlc: Plotly Candlestick object.
    """ # noqa
    # Create the candlestick chart
    fig = go.Ohlc(x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"])  # noqa

    try:
        # Set the color for increasing (green) and decreasing (red) candles
        fig.increasing.fillcolor = '#00FF00'
        fig.decreasing.fillcolor = '#FF0000'
        fig.increasing.line.color = '#00FF00'
        fig.decreasing.line.color = '#FF0000'
    except ValueError:
        # Set the color for increasing (green) and decreasing (red) candles
        # fig.increasing.line.color = '#00FF00'
        # fig.decreasing.line.color = '#FF0000'

        # Set the color for increasing and decreasing candles to blue (converted to white in grayscale image)
        fig.increasing.line.color = '#0000FF'
        fig.decreasing.line.color = '#0000FF'
    return fig


def create_line(df) -> go.Scatter:
    """
    Creates a line chart for the given DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing the bond index data with columns Date, Index.

    Returns:
    go.Scatter: Plotly Scatter object.
    """  # noqa
    # Create the line chart
    fig = go.Scatter(x=df["Date"], y=df["Index"], mode="lines")

    # Set the line color to white
    fig.line.color = '#0000FF'

    return fig


def make_figure(equity_df, currency_df, bond_df) -> go.Figure:
    """
    Creates a Plotly figure with subplots for equity, currency, and bond data.

    Parameters:
    equity_df (pd.DataFrame): DataFrame containing the equity data.
    currency_df (pd.DataFrame): DataFrame containing the currency data.
    bond_df (pd.DataFrame): DataFrame containing the bond index data.

    Returns:
    go.Figure: Plotly Figure object.
    """
    equity_df['Date'] = pd.to_datetime(equity_df['Date'], format='%d%b%Y:%H:%M:%S.%f')
    currency_df['Date'] = pd.to_datetime(currency_df['Date'], format='%d%b%Y:%H:%M:%S.%f')
    bond_df['Date'] = pd.to_datetime(bond_df['Date'], format='%d%b%Y:%H:%M:%S.%f')

    # Create a subplot figure with 3 rows and 1 column
    fig = make_subplots(rows=3, cols=1,
                        shared_xaxes=True,
                        shared_yaxes=False,
                        vertical_spacing=0.1)  # noqa

    # Update the layout of the figure
    fig.update_layout(plot_bgcolor='black', font_color="white", autosize=True, width=1000, height=1000)   # noqa
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), paper_bgcolor="black")

    # Disable the legend
    fig.update_layout(showlegend=False)

    for i in range(1, 4):
        fig.update_xaxes(row=i, col=1, rangeslider_visible=False)
        # Update x-axis and y-axis properties for each subplot
        fig.update_xaxes(row=i, col=1, showgrid=False, zeroline=False, showline=False, gridcolor="white", showticklabels=False, linecolor="white")    # noqa
        fig.update_yaxes(row=i, col=1, showgrid=False, zeroline=False, showline=False, gridcolor="white", showticklabels=False, linecolor="white")    # noqa

    # Add the candlestick and line charts to the subplots
    fig.add_trace(create_candlestick(equity_df), row=1, col=1)
    fig.add_trace(create_candlestick(currency_df), row=2, col=1)
    fig.add_trace(create_line(bond_df), row=3, col=1)

    fig.update_xaxes(type="category")

    return fig


def create_image(equity_df, currency_df, bond_df, width=64, height=64) -> np.ndarray:   # noqa
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
    """ # noqa
    # Create the Plotly figure
    fig: go.Figure = make_figure(equity_df, currency_df, bond_df)

    # Save the figure to a BytesIO object as a PNG image
    img_bytes = BytesIO()
    fig.write_image(img_bytes, format="png",
                    width=width, height=height)
    img_bytes.seek(0)

    # Open the image using PIL
    image = Image.open(img_bytes).resize((width, height), resample=Resampling.NEAREST)

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
