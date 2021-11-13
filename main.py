from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import time

MODEL_SAVE_LOCATION = "mnist_cnn.pt"

def current_milli_time():
    return round(time.time() * 1000)

def Net():
    layers = []
    layers.append(nn.Conv2d(1, 32, (3,3), 1))
    layers.append(nn.ReLU())
    layers.append(nn.Conv2d(32, 64, (3,3), 1))
    layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(2))
    layers.append(nn.Dropout(0.25))
    layers.append(nn.Flatten(1))
    layers.append(nn.Linear(9216, 128))
    layers.append(nn.Dropout(0.5))
    layers.append(nn.Linear(128, 10))
    layers.append(nn.LogSoftmax(1))
    return nn.Sequential(*layers)

def train(model, device, train_loader, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                0, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    torch.manual_seed(current_milli_time())
    device = torch.device("cpu")
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('../data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('../data', train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset1, 64)
    test_loader = torch.utils.data.DataLoader(dataset2, 1000)

    model = Net().to(device)
    print(model)
    print()
    print()
    optimizer = optim.Adadelta(model.parameters(), lr = 1.0)

    try:
        model.load_state_dict(torch.load(MODEL_SAVE_LOCATION))
        print("Loaded Model from file '" + MODEL_SAVE_LOCATION + "'")
    except FileNotFoundError as e:
        print("Could not load file '" + MODEL_SAVE_LOCATION + "'")

    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

    for epoch in range(1, 10):
        train(model, device, train_loader, optimizer)
        test(model, device, test_loader)
        scheduler.step()
        torch.save(model.state_dict(), MODEL_SAVE_LOCATION)
        print("Saving was successfull!")
        print()

if __name__ == '__main__':
    main()






