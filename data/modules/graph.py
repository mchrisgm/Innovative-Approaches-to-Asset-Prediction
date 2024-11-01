import pandas as pd
import numpy as np
from PIL import Image
import plotly.graph_objects as go
from io import BytesIO
from plotly.subplots import make_subplots

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def create_candlestick(df):
    return go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        increasing_line_color='#00FF00',
        decreasing_line_color='#FF0000',
        line_width=1
    )

def create_ohlc(df):
    return go.Ohlc(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close']
    )

def create_line(x, y):
    return go.Scatter(
        x=x,
        y=y,
        mode='lines',
        line=dict(color='#FFFFFF', width=1),
    )

def make_figure(data_frames, lookback=5):
    num_frames = len(data_frames) + 0
    fig = make_subplots(rows=num_frames, cols=1, shared_xaxes=True, vertical_spacing=0.02)

    for i, df in enumerate(data_frames, 1):
        # Calculate RSI using the full dataset
        rsi = calculate_rsi(df['Close'])

        # Select only the last 'lookback' days for plotting
        plot_start = df.index[-lookback]
        df_plot = df.loc[plot_start:]
        rsi_plot = rsi.loc[plot_start:]

        fig.add_trace(create_candlestick(df_plot), row=i, col=1)
        # fig.add_trace(create_ohlc(df_plot), row=i, col=1)
        # fig.add_trace(create_line(rsi_plot.index, rsi_plot), row=num_frames + 1, col=1)

    fig.update_layout(showlegend=False)
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.update_layout(plot_bgcolor='black', font_color="white", autosize=True, width=120, height=120)
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), paper_bgcolor="black")

    fig.update_xaxes(type="category")

    for i in range(1, num_frames + 2):
        fig.update_xaxes(row=i, col=1, showgrid=False, zeroline=False, showline=False, gridcolor="white", showticklabels=False, linecolor="white")
        fig.update_yaxes(row=i, col=1, showgrid=False, zeroline=False, showline=False, gridcolor="white", showticklabels=False, linecolor="white")

    return fig


def create_image(data_frames, width=128, height=128, lookback=5, rgb_channels=1):
    for df in data_frames:
        if len(df) < lookback * 4:
            Exception(f"DataFrame must have at least {lookback * 4} rows. Current length: {len(df)}")
        # df['Datetime'] = pd.to_datetime(df['Date'])
        df["Date"].apply(lambda x: x.strftime('%Y-%m-%d'))
        df.set_index('Date', inplace=True)
    
    fig = make_figure(data_frames, lookback)

    img_bytes = BytesIO()
    fig.write_image(img_bytes, format="png", width=width, height=height, scale=1)
    img_bytes.seek(0)

    rgb_mode = {
        1: "L",
        2: "CMYK",
        3: "RGB",
        4: "RGBA"
    }

    image = Image.open(img_bytes).convert(rgb_mode.get(rgb_channels, "L"))
    image = image.point(lambda p: p > 0 and 255)
    image_array = np.array(image)

    return image_array

if __name__ == "__main__":
    np.random.seed(0)
    lookback = 5
    date_period = lookback * 4  # Each DataFrame has 4 times the length of lookback
    dates = pd.date_range(start="2020-01-01", periods=date_period)

    def create_sample_df():
        return pd.DataFrame({
            "Date": dates,
            "Open": np.random.randint(100, 200, date_period),
            "High": np.random.randint(200, 300, date_period),
            "Low": np.random.randint(50, 100, date_period),
            "Close": np.random.randint(100, 200, date_period)
        })

    data_frames = [create_sample_df() for _ in range(1)]  # Create 3 sample dataframes

    img = create_image(data_frames, width=128, height=128, lookback=lookback, rgb_channels=3)
    print(img.shape)

    pil_image = Image.fromarray(img)
    pil_image.show()
    pil_image.save("./data/sample_image_generation.png")