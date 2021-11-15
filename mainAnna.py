#Different approach for setting up the neuronal network, based on the video from Akash Bhiwgade


import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

#dirty fix for OpenMP error
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#parameters
num_epochs = 10
num_classes = 10
learning_rate = 0.001
batch_size = 10
input_size = 784 #28*28
hidden_layers = 50


#get dataset
train_data = torchvision.datasets.MNIST(root = "./dataset", train = True, transform = transforms.ToTensor(), download = True)
test_data = torchvision.datasets.MNIST(root = "./dataset", train = False, transform = transforms.ToTensor(), download = True)

train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True, num_workers = 0)
test_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size = batch_size, shuffle = False, num_workers = 0)


#code snippet to show pictures and corresponding value

#checkdata = iter(train_loader)
#img, lab = next(checkdata)
#print(img.shape, lab.shape)

#for i in range(9):
#    print(lab[i])
#    plt.subplot(3,3, i+1)
#    plt.imshow(img[i][0], cmap = "gray")
#plt.show()

#init neuronal network with three layers (this seems to be considered deep)
class DigitRecognizer(nn.Module):

    def __init__(self, input_size, hidden_layers, num_classes):
        
        super(DigitRecognizer, self).__init__()
        self.input = nn.Linear(in_features = input_size, out_features = hidden_layers)
        self.relu1 = nn.ReLU()
        self.hidden1 = nn.Linear(in_features = hidden_layers, out_features = hidden_layers)
        self.relu2 = nn.ReLU()
        self.hidden2 = nn.Linear(in_features = hidden_layers, out_features = hidden_layers)
        self.relu3 = nn.ReLU()
        self.hidden3 = nn.Linear(in_features = hidden_layers, out_features = hidden_layers)
        self.relu4 = nn.ReLU()
        self.output = nn.Linear(in_features = hidden_layers, out_features = num_classes)

    def forward(self, X):
        model = self.input(X)
        model = self.relu1(model)
        model = self.hidden1(model)
        model = self.relu2(model)
        model = self.hidden2(model)
        model = self.relu3(model)
        model = self.hidden3(model)
        model = self.relu4(model)
        model = self.output(model)

        return model

model = DigitRecognizer(input_size, hidden_layers, num_classes)
repr(model)

criteria = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

#code for training
samples = len(train_loader)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

for epoch in range(num_epochs):
    for step, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criteria(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("Epoch: {}/{}, step: {}/{}, loss: {:.4f}".format(epoch, num_epochs, step, samples, loss.item()))

