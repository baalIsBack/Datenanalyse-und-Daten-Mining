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
    layers.append(nn.Conv2d(1, 32, (3,3)))
    layers.append(nn.ReLU())
    layers.append(nn.Conv2d(32, 64, (3,3)))
    layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(2))
    layers.append(nn.Dropout(0.25))
    layers.append(nn.Flatten(1))
    layers.append(nn.Linear(9216, 128))
    layers.append(nn.Dropout(0.5))
    layers.append(nn.Linear(128, 10))
    layers.append(nn.LogSoftmax(1))
    return nn.Sequential(*layers)

def Net2():
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

def train(model, device, loader, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(loader):
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        loss = F.nll_loss(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('{:.0f}%({}/{}) \tLoss: {:.8f}'.format(
                100. * batch_idx / len(loader),
                batch_idx * len(data),
                len(loader.dataset),
                loss.item()
            ))


def test(model, device, loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            prediction = output.argmax(dim=1, keepdim=True)#index of max log-probability
            correct += prediction.eq(target.view_as(prediction)).sum().item()

    test_loss /= len(loader.dataset)

    print('Average_loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss,
        correct,
        len(loader.dataset),
        100. * correct / len(loader.dataset)
    ))


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
    optimizer = optim.Adadelta(model.parameters())#TODO change learning algorithm to backprop maybe?

    try:
        model.load_state_dict(torch.load(MODEL_SAVE_LOCATION))
        print("Loaded Model from file '" + MODEL_SAVE_LOCATION + "'")
    except FileNotFoundError as e:
        print("Could not load file '" + MODEL_SAVE_LOCATION + "'")

    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

    for epoch in range(1, 10):
        print("Entering epoch " + str(epoch))
        train(model, device, train_loader, optimizer)
        test(model, device, test_loader)
        scheduler.step()
        torch.save(model.state_dict(), MODEL_SAVE_LOCATION)
        print("Saved successfully!")
        print()

if __name__ == '__main__':
    main()






