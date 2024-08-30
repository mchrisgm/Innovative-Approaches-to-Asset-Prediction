import torch
import torch.nn as nn
import torch.nn.functional as F

import json
import os

__all__ = ['FlexibleNet', 'save_model', 'StockPredictor']


class FlexibleNet(nn.Module):
    def __init__(self, config):
        super(FlexibleNet, self).__init__()
        layers = []
        in_channels = config['in_channels']

        # Define the convolutional layers with optional pooling
        for idx, ((out_channels, kernel_size, padding), pool_size) in enumerate(zip(config['conv_layers'], config['pool_layers'])):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(negative_slope=config['leak']))
            if pool_size:
                layers.append(nn.MaxPool2d(kernel_size=pool_size))
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)
        self.flatten = nn.Flatten()

        # Calculate the size of the flattened features after all convolutions and pooling layers
        with torch.no_grad():
            self.feature_dim = self._calculate_feature_dim(config['img_size'], config['in_channels'])

        # LSTM layer
        self.lstm = nn.LSTM(input_size=self.feature_dim, hidden_size=config['lstm_hidden_size'], num_layers=config['lstm_layers'], batch_first=True)

        fc_layers = []
        input_dim = config['lstm_hidden_size']
        for hidden_size in config['fc_layers']:
            fc_layers.append(nn.Linear(input_dim, hidden_size))
            fc_layers.append(nn.BatchNorm1d(hidden_size))
            fc_layers.append(nn.LeakyReLU(negative_slope=config['leak']))
            fc_layers.append(nn.Dropout(config['dropout']))
            input_dim = hidden_size

        fc_layers.append(nn.Linear(input_dim, config['output_size']))
        self.fc_layers = nn.Sequential(*fc_layers)

        # Regularization
        self.weight_decay = 0.00001  # L2 regularization

    def _calculate_feature_dim(self, img_size, in_channels):
        x = torch.randn(1, in_channels, *img_size)
        x = self.conv_layers(x)
        x = self.flatten(x)
        return x.shape[1]

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = x.view(batch_size, 1, -1)  # Adjust the input size for the LSTM
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Get the output of the last LSTM cell
        x = self.fc_layers(x)
        x = F.log_softmax(x, dim=1)
        return x


def save_model(model: FlexibleNet, config: dict, val_accuracy: float,
               run_id: int, path: str = './deep_learning/models/'):
    # Find the pth file from the same run_id
    files = os.listdir(path)
    for file in files:
        if str(run_id) in file:
            os.remove(os.path.join(path, file))

    torch.save(model.state_dict(),
               f'./deep_learning/models/{run_id}.' +
               f'{config["data_filename"]}.' +
               f'{config["output_size"]}.{val_accuracy*100:.0f}.pth')
    json.dump(config,
              open(f'./deep_learning/models/{run_id}.' +
                   f'{config["data_filename"]}.{config["output_size"]}.' +
                   f'{val_accuracy*100:.0f}.json', 'w'))


class StockPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, lstm_hidden_dim, output_dim, dropout_prob=0.3):
        super(StockPredictor, self).__init__()

        # Convolutional Layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm1d(128)  # Batch normalization after the second conv layer

        # LSTM Layer
        self.lstm = nn.LSTM(input_size=128, hidden_size=lstm_hidden_dim, num_layers=1, batch_first=True)

        # Fully Connected Layers
        self.fc1 = nn.Linear(lstm_hidden_dim * (input_dim // 2), hidden_dim1)  # Adjust input size based on pooling
        self.bn2 = nn.BatchNorm1d(hidden_dim1)  # Batch normalization for the first hidden layer
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)  # Second hidden layer
        self.bn3 = nn.BatchNorm1d(hidden_dim2)  # Batch normalization for the second hidden layer
        self.fc3 = nn.Linear(hidden_dim2, output_dim)  # Output layer

        self.relu = nn.ReLU()  # Activation function
        self.dropout = nn.Dropout(dropout_prob)  # Dropout layer for regularization

    def forward(self, x):
        # Reshape input for convolutional layers (batch_size, channels, input_dim)
        x = x.unsqueeze(1)  # Add channel dimension: (batch_size, 1, input_dim)

        # Convolutional layers with ReLU and Max Pooling
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.pool(out)
        out = self.bn1(out)

        # Permute for LSTM input: (batch_size, seq_len, input_size)
        out = out.permute(0, 2, 1)

        # LSTM layer
        out, _ = self.lstm(out)

        # Flatten the output from LSTM layers
        out = out.contiguous().view(out.size(0), -1)

        # Fully connected layers
        out = self.fc1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc3(out)  # Output layer
        return out
