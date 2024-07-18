import random
import numpy as np
import torch
import pandas as pd
from torchvision.transforms.functional import pad
from torchvision import transforms
from PIL import Image
import numbers
import matplotlib.pyplot as plt


__all__ = ['set_seed', 'get_dataset', 'train_val_test_split', 'split_features_targets', 'quantize_labels', 'transform', 'display_n_random']


def set_seed(seed):
    """
    Use this to set ALL the random seeds to a fixed value and take out any randomness from cuda kernels
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False  ##uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms. -
    torch.backends.cudnn.enabled = False


def get_dataset(filename, length=None) -> np.ndarray:
    """
    Get the dataset from the filename
    """
    full_data = np.load(f"./data/processed/{filename}/data.npy", allow_pickle=True)   # noqa

    full_data = pd.DataFrame(full_data)

    full_data = full_data.sample(n=length if length != None else len(full_data))

    full_data.dropna(inplace=True)

    data = full_data.to_numpy()

    return data


def train_val_test_split(data, train_size=0.7, val_size=0.2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_size = int(len(data) * train_size)
    val_size = int(len(data) * val_size)
    test_size = len(data) - train_size - val_size
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    return train_data, val_data, test_data


def split_features_targets(data) -> tuple[np.ndarray, np.ndarray]:
    features = data[:, 0]
    targets = np.asarray(data[:, 1:], dtype=np.int64)
    return features, targets


def quantize_labels(targets, num_classes=2):
    """
    Quantize targets from the range [-1, 1] to the integer range [0, num_classes-1].

    Args:
        targets (numpy.ndarray): Array of targets in the range [-1, 1].
        num_classes (int): Number of classes to quantize to.

    Returns:
        numpy.ndarray: Quantized targets in the integer range [0, num_classes-1].
    """
    # Scale the range from [-1, 1] to [0, 1]
    scaled_targets = (targets + 1) / 2

    # Quantize to the range [0, num_classes-1]
    quantized_targets = np.floor(scaled_targets * num_classes).astype(np.int64)

    # Ensure targets are within the range [0, num_classes-1]
    quantized_targets = np.clip(quantized_targets, 0, num_classes-1)

    return quantized_targets


def get_padding(image):
    w, h = image.size
    max_wh = np.max([w, h])
    h_padding = (max_wh - w) / 2
    v_padding = (max_wh - h) / 2
    l_pad = h_padding if h_padding % 1 == 0 else h_padding+0.5
    t_pad = v_padding if v_padding % 1 == 0 else v_padding+0.5
    r_pad = h_padding if h_padding % 1 == 0 else h_padding-0.5
    b_pad = v_padding if v_padding % 1 == 0 else v_padding-0.5
    padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))
    return padding


class SquarePad(object):
    def __init__(self, fill=0, padding_mode='constant'):
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        img = Image.fromarray(img)
        return pad(img, get_padding(img), self.fill, self.padding_mode)

    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1}, padding_mode={2})'.format(self.fill, self.padding_mode)


def transform(resize=(64, 64)):
    # Transformations for the images
    transform = transforms.Compose([
        SquarePad(),  # Make the image square by padding
        transforms.Resize(resize),  # Resize
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
    ])
    return transform


# helper function
def display_n_random(data, labels, n=8):
    '''
    Selects n random datapoints and their corresponding labels from a dataset
    '''
    assert len(data) == len(labels)

    perm = torch.randperm(len(data))
    images, labels = data[perm][:n], labels[perm][:n]

    # plot images
    plt.figure(figsize=(16, 6))
    for i in range(n):
        plt.subplot(2, 4, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title(labels[i])
        plt.axis('off')
    plt.show()