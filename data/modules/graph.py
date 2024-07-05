from io import BytesIO
import numpy as np
import pandas as pd
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_candlestick(df) -> go.Candlestick:
    fig = go.Candlestick(x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"])
    # Set line and fill colors
    fig.increasing.fillcolor = '#00FF00'
    fig.decreasing.fillcolor = '#FF0000'
    fig.increasing.line.color = '#00FF00'
    fig.decreasing.line.color = '#FF0000'
    return fig


def create_line(df) -> go.Scatter:
    fig = go.Scatter(x=df["Date"], y=df["Index"], mode="lines")
    # Set line color
    fig.line.color = '#FFFFFF'
    return fig


def make_figure(equity_df, currency_df, bond_df) -> go.Figure:
    fig = make_subplots(rows=3, cols=1,
                        shared_xaxes=True, shared_yaxes=False,
                        vertical_spacing=0)

    fig.update_layout(plot_bgcolor='black',  font_color="white", autosize=True, width=100, height=100) # noqa
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), paper_bgcolor="black")
    fig.update_layout(xaxis=dict(type="category"))

    fig.update_xaxes(showgrid=False, zeroline=False,
                     showline=False, gridcolor="white",
                     showticklabels=False, linecolor="white")
    fig.update_yaxes(showgrid=False, zeroline=False,
                     showline=False, gridcolor="white",
                     showticklabels=False, linecolor="white")

    for i in range(1, 4):
        fig.update_xaxes(row=i, col=1, rangeslider_visible=False)

    fig.update_layout(showlegend=False)

    fig.add_trace(create_candlestick(equity_df), row=1, col=1)
    fig.add_trace(create_candlestick(currency_df), row=2, col=1)
    fig.add_trace(create_line(bond_df), row=3, col=1)
    return fig


def create_image(equity_df, currency_df, bond_df, width=64, height=64) -> np.ndarray:
    fig: go.Figure = make_figure(equity_df, currency_df, bond_df)
    img_bytes = BytesIO()
    fig.write_image(img_bytes, format="png", width=width, height=height)
    img_bytes.seek(0)
    # Convert image to grayscale numpy array
    image = Image.open(img_bytes)
    # image.show()
    image_array = np.array(image)
    # Custom conversion to grayscale
    grayscale_image_array = np.dot(image_array[..., :3], [0.2, 0.8, 1])
    grayscale_image_array = grayscale_image_array.astype(np.uint8)
    return grayscale_image_array


if __name__ == "__main__":
    # Create sample randomized data
    date_period = 5
    dates = pd.date_range(start="2020-01-01", periods=date_period)
    equity_df = pd.DataFrame({
        "Date": dates,
        "Open": np.random.randint(100, 200, date_period),
        "High": np.random.randint(200, 300, date_period),
        "Low": np.random.randint(50, 100, date_period),
        "Close": np.random.randint(100, 200, date_period)
    })
    currency_df = pd.DataFrame({
        "Date": dates,
        "Open": np.random.randint(100, 200, date_period),
        "High": np.random.randint(200, 300, date_period),
        "Low": np.random.randint(50, 100, date_period),
        "Close": np.random.randint(100, 200, date_period)
    })
    bond_df = pd.DataFrame({
        "Date": dates,
        "Index": np.random.randint(100, 200, date_period)
    })
    img = create_image(equity_df, currency_df, bond_df)
    print(img.shape)
    pil_image = Image.fromarray(img)
    pil_image.show()
    pil_image.save("./data/sample_image_generation.png")
