import pandas as pd
import numpy as np
from PIL import Image, ImageFilter
import plotly.graph_objects as go
from io import BytesIO
from plotly.subplots import make_subplots

# Exported functions
__all__ = ["create_image"]

def create_candlestick(df):
    """
    Creates a candlestick chart for the DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing the stock data with columns Date, Open, High, Low, Close.
    """
    assert "Date" in df.columns, "Date column not found in the DataFrame"
    assert "Open" in df.columns, "Open column not found in the DataFrame"
    assert "High" in df.columns, "High column not found in the DataFrame"
    assert "Low" in df.columns, "Low column not found in the DataFrame"
    assert "Close" in df.columns, "Close column not found in the DataFrame"
    
    # f = go.Ohlc(
    #         x=df['Date'],
    #         open=df['Open'],
    #         high=df['High'],
    #         low=df['Low'],
    #         close=df['Close']
    #     )
    # f.tickwidth = 0.5
    # f.increasing.line.color = '#00FF00'
    # f.decreasing.line.color = '#FF0000'
    
    f = go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            increasing_line_color='#00FF00',
            decreasing_line_color='#FF0000',
            line_width=1
        )
    f.whiskerwidth = 0
    return f

def create_line(df):
    """
    Creates a line chart for the DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing the bond index data with columns Date, Index.
    """
    assert "Date" in df.columns, "Date column not found in the DataFrame"
    assert "Index" in df.columns, "Index column not found in the DataFrame"
    
    f = go.Scatter(
            x=df['Date'],
            y=df['Index'],
            mode='lines',
            line=dict(color='#0000FF', width=1),
        )
    return f

def make_figure(equity_df=None, currency_df=None, bond_df=None, lookback=5):
    """
    Creates a Plotly figure with subplots for equity, currency, and bond data.

    Parameters:
    equity_df (pd.DataFrame): DataFrame containing the equity data.
    currency_df (pd.DataFrame): DataFrame containing the currency data.
    bond_df (pd.DataFrame): DataFrame containing the bond index data.

    Returns:
    plotly.graph_objects.Figure: Plotly Figure object.
    """
    rows = sum([1 for df in [equity_df, currency_df, bond_df] if df is not None])
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    fig.update_layout(paper_bgcolor='black', plot_bgcolor='black')

    row = 1
    if equity_df is not None:
        equity_df['Date'] = pd.to_datetime(equity_df['Date'], format='%d%b%Y:%H:%M:%S.%f')
        fig.add_trace(create_candlestick(equity_df), row=row, col=1)
        row += 1

    if currency_df is not None:
        currency_df['Date'] = pd.to_datetime(currency_df['Date'], format='%d%b%Y:%H:%M:%S.%f')
        fig.add_trace(create_candlestick(currency_df), row=row, col=1)
        row += 1

    if bond_df is not None:
        bond_df['Date'] = pd.to_datetime(bond_df['Date'], format='%d%b%Y:%H:%M:%S.%f')
        fig.add_trace(create_line(bond_df), row=row, col=1)

    # Update the layout of the figure
    fig.update_layout(plot_bgcolor='black', font_color="white", autosize=False, width=lookback*3, height=32)   # noqa
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), paper_bgcolor="black")

    # Disable the legend
    fig.update_layout(showlegend=False)

    for i in range(1, rows + 1):
        fig.update_xaxes(row=i, col=1, rangeslider_visible=False)
        # Update x-axis and y-axis properties for each subplot
        fig.update_xaxes(row=i, col=1, showgrid=False, zeroline=False, showline=False, gridcolor="white", showticklabels=False, linecolor="white")    # noqa
        fig.update_yaxes(row=i, col=1, showgrid=False, zeroline=False, showline=False, gridcolor="white", showticklabels=False, linecolor="white")    # noqa

    fig.update_xaxes(type="category")
    return fig

def create_image(equity_df=None, currency_df=None, bond_df=None, width=128, height=128, lookback=5, rgb_channels=1):
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
    # Create the Plotly figure
    fig = make_figure(equity_df, currency_df, bond_df, lookback=lookback)

    # Save the figure to a BytesIO object as a PNG image
    img_bytes = BytesIO()
    fig.write_image(img_bytes, format="png", width=width, height=height, scale=1)
    img_bytes.seek(0)

    rgb_mode = {
        1: "L",
        2: "CMYK",
        3: "RGB",
        4: "RGBA"
    }

    # Open the image using PIL
    image = Image.open(img_bytes).convert(rgb_mode.get(rgb_channels, "L"))

    # image = image.filter(ImageFilter.SHARPEN)

    # Set all pixels to white (255) if the pixel value is greater than 1
    image = image.point(lambda p: p > 1 and 255)

    # Convert the image to a numpy array
    image_array = np.array(image)

    return image_array

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
    img = create_image(equity_df, currency_df, bond_df,
                       width=128, height=128, lookback=5,
                       rgb_channels=3)
    print(img.shape)

    # Display and save the image
    pil_image = Image.fromarray(img)
    pil_image.show()
    pil_image.save("./data/sample_image_generation.png")
