import torch
import torch.nn as nn

import json
import os

__all__ = ['FlexibleNet', 'save_model']


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
