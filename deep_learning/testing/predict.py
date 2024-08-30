import json
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from deep_learning import FlexibleNet, StockPredictor

import numpy as np
import pandas as pd

from data import create_image
from deep_learning import ImageDataset, transform, feature_eng
from deep_learning import dequantize_labels

__all__ = ['predict_cnn', 'predict_lstm']


def predict_cnn(model_filename: str, dataframes: list[pd.DataFrame], device='cpu') -> list[float]:
    config = json.load(open(f'./deep_learning/models/{model_filename}.json'))
    model = FlexibleNet(config).to(device)
    model.load_state_dict(torch.load(f'./deep_learning/models/{model_filename}.pth', map_location=device))
    model.eval()

    outs = []
    with torch.no_grad():
        for dataframe in dataframes:
            # Create image from dataframe
            image = create_image(equity_df=dataframe, bond_df=None, currency_df=None,
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


def predict_lstm(model_filename: str, df: pd.DataFrame, device='cpu') -> list[float]:
    # Ensure the model is in evaluation mode
    hidden_dim1 = 128  # Number of neurons in the first hidden layer
    hidden_dim2 = 32   # Number of neurons in the second hidden layer
    lstm_hidden_dim = 128  # Number of LSTM hidden units
    input_dim = 9
    output_dim = 2     # Binary output: Long, Short

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StockPredictor(input_dim, hidden_dim1, hidden_dim2, lstm_hidden_dim, output_dim).to(device)
    model.load_state_dict(torch.load(model_filename))
    model.eval()

    scaler = StandardScaler()
    
    # Preprocessing: Apply the same feature engineering used during training
    df = feature_eng(df)
    df = df.tail(5)
    
    # Select the relevant features (make sure to drop any columns not used during training)
    features = df.drop(columns=['Label', 'Date'], errors='ignore')  # Adjust if needed

    # Normalize the data using the scaler fitted during training
    features_scaled = scaler.fit_transform(features)

    # Convert the features to a PyTorch tensor
    features_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(device)
    
    # Add batch dimension (batch_size, input_dim)
    features_tensor = features_tensor.unsqueeze(0) if len(features_tensor.shape) == 1 else features_tensor
    
    # Perform model inference
    with torch.no_grad():
        outputs = model(features_tensor)

        _, predicted_class = torch.max(outputs, 1)
        predicted_class = predicted_class[-1]
    
    # Map numeric prediction to class labels
    class_mapping = {0: -1, 1: 1}
    predicted_class = class_mapping[predicted_class.item()]

    return predicted_class
