from io import BytesIO
import numpy as np
import pandas as pd
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_candlestick(df):
    fig = go.Figure(data=[go.Candlestick(x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"])])

    # fig.update_layout(xaxis_rangeslider_visible=False)
    # fig.update_layout(plot_bgcolor='black',  font_color="white", autosize=True, width=120, height=120) # noqa
    # fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), paper_bgcolor="black")
    # # fig.update_layout(xaxis=dict(type="category"))
    # fig.update_xaxes(showgrid=False, zeroline=False, showline=False, gridcolor="white", showticklabels=False, linecolor="white")
    # fig.update_yaxes(showgrid=False, zeroline=False, showline=False, gridcolor="white", showticklabels=False, linecolor="white")
    return fig


def make_figure(df) -> go.Figure:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, shared_yaxes=False)
    return fig


def create_image(df) -> np.ndarray:
    fig: go.Figure = make_figure(df)
    img = BytesIO()
    fig.write_image(img, format="png", width=32, height=32)
    img.seek(0)
    image = Image.open(img)
    image_array = np.array(image)
    return image_array


if __name__ == "__main__":
    df = pd.DataFrame({
        "Date": ["2021-01-01", "2021-01-02", "2021-01-03"],
        "Open": [100, 200, 300],
        "High": [150, 250, 350],
        "Low": [50, 150, 250],
        "Close": [120, 220, 320]
    })
    img = create_image(df)
    print(img.shape)
