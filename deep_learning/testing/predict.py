import json
import torch
from torch.utils.data import DataLoader
from deep_learning import FlexibleNet

import numpy as np
import pandas as pd

from data import create_image
from deep_learning import ImageDataset, transform
from deep_learning import dequantize_labels

__all__ = ['predict']


def predict(model_filename: str, dataframes: list[pd.DataFrame], device='cpu') -> list[float]:
    config = json.load(open(f'./deep_learning/models/{model_filename}.json'))
    model = FlexibleNet(config).to(device)
    model.load_state_dict(torch.load(f'./deep_learning/models/{model_filename}.pth', map_location=device))
    model.eval()

    outs = []
    with torch.no_grad():
        for dataframe in dataframes:
            # Create image from dataframe
            image = create_image([dataframe],
                                 width=config["img_size"][0], height=config["img_size"][1],
                                 lookback=len(dataframe), rgb_channels=config["in_channels"])

            # image = image.astype(np.float32)
            # Create dataset and dataloader
            trform = transform((64, 64))
            dataset = ImageDataset([image], [0.0], transform=trform)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

            # Predict
            for batch in dataloader:
                images, targets = batch
                images, targets = images.to(device), targets.squeeze().to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                outs.append(predicted.item())

    # Dequantize the outputs
    outs = dequantize_labels(np.array(outs), config['output_size'])
    return outs
