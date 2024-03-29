import torch
import torch.nn as nn
import torch.nn.functional
import torch.optim
import time
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from torchvision.datasets import MNIST

# definition of Parameters
numberOfEpochs = 3
batchSize = 12
learningrate = 0.001 #Bei Optimizer Adam 0.001, bei SGD 0.01

# definition of the Neural Network Model with 4 Layers
def NeuralNetwork():
    layers=[]
    layers.append(nn.Conv2d(1, 16, 3, 1))
    layers.append(nn.ReLU())
    layers.append(nn.Flatten(1))
    layers.append(nn.Linear(26*26*16, 128))
    layers.append(nn.ReLU())
    layers.append(nn.Dropout(0.25))
    layers.append(nn.Linear(128,64))
    layers.append(nn.ReLU())
    layers.append(nn.Dropout(0.5))
    layers.append(nn.Linear(64, 10))
    return nn.Sequential(*layers)

#training for a single Epoch
def train(model, device, train_loader, optimizer,epoch):
    print(" Start of Training Epoch", epoch, ": ")

    for batch, (image, target) in enumerate(train_loader):
        image, target = image.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(image)
        loss = nn.CrossEntropyLoss()(output,target)
        loss.backward()
        optimizer.step()
        if batch % 10 == 0:
            print(' Progress: {:.0f}% with loss: {:.6f}'.format
                (
                100. * batch / len(train_loader), loss.item())
                )


#testing Accuracy after Training Cylce for an Epoch
def test(model, device, test_loader,epoch):
    correctly_classified = 0
    total = 0
    with torch.no_grad():
        for image, target in test_loader:
            image, target= image.to(device),target.to(device)
            outputs = model(image)
            _,predicted = torch.max(outputs, 1)
            total += target.size(0)
            correctly_classified += (predicted == target).sum().item()
    accuracy=100.0 * correctly_classified / total
    print(' Accuracy in Epoch {}: {:.0f}% \n'.format(
        epoch,
        accuracy
        ))


#Seed
def current_milli_time():
    return round(time.time() * 1000)

def main():
    #Seed setting
    torch.manual_seed(current_milli_time())
    #check if GPU is available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])

    #load MNIST dataset
    dataset_training = MNIST(root='./rootMNIST', train=True,download=True, transform=transform)

    dataset_testing =MNIST(root='./rootMNIST', train=False,download=True, transform=transform)

    train_loader = DataLoader(dataset_training, batch_size=batchSize,shuffle=True)

    test_loader = DataLoader(dataset_testing, batch_size=batchSize,shuffle=False)


    print(" Number of Elements in Training Dataset:",len(dataset_training))
    print(" Number of Elements in Test Dataset:",len(dataset_testing))
    model = NeuralNetwork().to(device)
    print(" Model:", model)

    #initialisation of Gradient Descent as optimizer and  StepLR as scheduler
    optimizer = torch.optim.AdamW(model.parameters(), learningrate)
    scheduler = StepLR(optimizer, step_size=1)

    #loop over defined number of Epochs and perform training and testing for each cycle
    for epoch in range(1, numberOfEpochs + 1):
        train( model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader,epoch)
        #decreases learning rate by 0.1 after each epoch
        scheduler.step()



if __name__ == '__main__':
    main()
