import json
import torch
import torch.nn as nn
from deep_learning import FlexibleNet


__all__ = ['test']


# Load the best model
# filename = f'{run_id}.{config["data_filename"]}.{config["output_size"]}.{val_accuracy*100:.0f}'


def test(filename: str, testloader: torch.utils.data.DataLoader, device: torch.device) -> tuple[float, float]:
    config = json.load(open(f'./deep_learning/models/{filename}.json'))
    model = FlexibleNet(config).to(device)
    model.load_state_dict(torch.load(f'./deep_learning/models/{filename}.pth'))

    criterion = nn.CrossEntropyLoss()

    # Test the model
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in testloader:
            images, targets = images.to(device), targets.squeeze(1).to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    test_loss = test_loss / len(testloader.dataset)
    test_accuracy = correct / total

    # print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    return test_loss, test_accuracy
