import json
import torch
from deep_learning import FlexibleNet
import pandas as pd

from data import create_image

__all__ = ['predict']


def predict(model_filename: str, dataframes: list[pd.DataFrame], device='cpu') -> list[float]:
    config = json.load(open(f'./deep_learning/models/{model_filename}.json'))
    model = FlexibleNet(config).to(device)
    model.load_state_dict(torch.load(f'./deep_learning/models/{model_filename}.pth'))

    # Predict the data
    model.eval()

    with torch.no_grad():
        outs = []
        for dataframe in dataframes:
            # Preprocess the data
            image = create_image(equity_df=dataframe, bond_df=None, currency_df=None,
                                 width=config["img_size"][0], height=config["img_size"][1],
                                 lookback=len(dataframe), rgb_channels=config["in_channels"])  # noqa

            image = torch.tensor(image).unsqueeze(0).to(device)
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            outs.append(predicted.item())

    return outs
