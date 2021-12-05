import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional
import torch.optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from torchvision.datasets import MNIST
from tqdm import tqdm



# definition of Parameters
numberOfEpochs = 3
batchSize = 12
learningrate = 0.001
numberOfClients = 10
numberOfSelectedClients = 4
numberOfRounds = 20

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

#testing accuracy after each round
def test(model, device, test_loader):
    correctly_classified = 0
    loss = 0
    with torch.no_grad():
        for image, target in test_loader:
            image, target= image.to(device),target.to(device)
            outputs = model(image)
            _,predicted = torch.max(outputs, 1)
            loss +=nn.CrossEntropyLoss()(outputs,predicted).item()
            correctly_classified += (predicted == target).sum().item()

    loss /= len(test_loader.dataset)
    accuracy = 100.0 *correctly_classified / len(test_loader.dataset)

    return loss, accuracy

#trains model of client on client data
def train_client(model, device,optimizer, train_loader, epoch):
     for epochX in range(epoch):
        for batch, (image, target) in enumerate(train_loader):
            image, target = image.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(image)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer.step()
     return loss.item()

# aggregates  results of the client model and takes the mean of the weights
def server_aggregate(global_model, client_models):
    #get state of global model
    update_global_model = global_model.state_dict()
    #average the weights of clients and update global model
    for weighti in update_global_model.keys():
      update_global_model[weighti] = torch.stack([client_models[i].state_dict()[weighti].float() for i in range(len(client_models))], 0).mean(0)
      global_model.load_state_dict(update_global_model)
    #update the models of clients before next training
    for model in client_models:
          model.load_state_dict(global_model.state_dict())

def main():
    torch.manual_seed(np.round(time.time() * 1000))

    #check if GPU is available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])

    #load MNIST dataset
    dataset_training = MNIST(root='./rootMNIST', train=True,download=True, transform=transform)

    dataset_testing =MNIST(root='./rootMNIST', train=False,download=True, transform=transform)

    # Dividing the training data into num_clients, with each client having equal number of images
    partition_of_training_data = torch.utils.data.random_split(dataset_training, [int(dataset_training.data.shape[0] / numberOfClients) for _ in
                                                                                  range(numberOfClients)])


    train_loader = [torch.utils.data.DataLoader(partX, batch_size=batchSize, shuffle=True) for partX  in partition_of_training_data]

    test_loader = DataLoader(dataset_testing, batch_size=batchSize,shuffle=False)


    print(" Number of Elements in Training Dataset:",len(dataset_training))
    print(" Number of Elements in Test Dataset:",len(dataset_testing))
    #initialize global server model and identical client models
    global_model = NeuralNetwork().to(device)
    client_models = [NeuralNetwork().to(device) for _ in range(numberOfSelectedClients)]
    print(" Model:", global_model)

    for model in client_models:
        model.load_state_dict(global_model.state_dict())
    # initialisation of Stochastic Gradient Descent as optimizer and  StepLR as scheduler for all client models
    optimizer = [torch.optim.AdamW(model.parameters(), learningrate) for model in client_models]
    scheduler = [StepLR(optim, step_size=1)for optim in optimizer]

    #waiting 0.1s so that prints can finish before tqdm
    time.sleep(0.1)

    for round in range(numberOfRounds):
        # select random clients
        selectedClients = np.random.permutation(numberOfClients)[:numberOfSelectedClients]

        # train selected clients
        current_round_loss = 0
        for i in tqdm(range(numberOfSelectedClients),position=0,leave=True):
            current_round_loss += train_client(client_models[i], device,optimizer[i], train_loader[selectedClients[i]], epoch=numberOfEpochs)

        # aggregate results of client training and update global and client models
        server_aggregate(global_model, client_models)

        #test current state of updated global model
        test_loss, accuracy = test(global_model, device,test_loader)
        print('Round %d :' % (round+1), 'Average loss during Training: %0.3g | Test loss: %0.3g | Accuracy: %0.3f' % (current_round_loss / numberOfSelectedClients, test_loss, accuracy))

        #adapt learning rates with scheduler
        for schedulerX in scheduler:
            schedulerX.step()

        #wait 0.1s so that prints can finish
        time.sleep(0.1)

if __name__ == '__main__':
    main()
