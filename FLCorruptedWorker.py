import sys
import time
import threading

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional
import torch.optim
from geom_median.torch import compute_geometric_median
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from torchvision.datasets import MNIST
from tqdm import tqdm
from pprint import pprint
from torch.utils.data import Dataset
import random as rand
#import skimage
#from openpyxl import *
#from openpyxl.utils import get_column_letter

# definition of Parameters
numberOfEpochs = 3
batchSize = 12
learningrate = 0.001
numberOfClients = 10
numberOfSelectedClients = 4
numberOfRounds = 20
# documentation
path = "Evaluation/FLCorruptedWorker/"

def getOther(n):
    x = np.random.randint(10)
    if x == n:
        return getOther(n)
    return x

class Dataset_RandomOther(Dataset):
    def __init__(self, transform):
        self.mnist = MNIST(root='./rootMNIST', train=True, download=True, transform=transform)

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        it = list(self.mnist[idx])
        it[-1] = getOther(it[-1])
        return it

class Dataset_Shift(Dataset):
    def __init__(self, transform):
        self.mnist = MNIST(root='./rootMNIST', train=True, download=True, transform=transform)

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        it = list(self.mnist[idx])
        it[-1] = it[-1]+1 % 10
        return it



# definition of the Neural Network Model with 4 Layers
def NeuralNetwork():
    layers = []
    layers.append(nn.Conv2d(1, 16, 3, 1))
    layers.append(nn.ReLU())
    layers.append(nn.Flatten(1))
    layers.append(nn.Linear(26 * 26 * 16, 128))
    layers.append(nn.ReLU())
    layers.append(nn.Dropout(0.25))
    layers.append(nn.Linear(128, 64))
    layers.append(nn.ReLU())
    layers.append(nn.Dropout(0.5))
    layers.append(nn.Linear(64, 10))
    return nn.Sequential(*layers)


# testing accuracy on global model after each round
def test(model, device, test_loader):
    correctly_classified = 0
    loss = 0
    with torch.no_grad():
        for image, target in test_loader:
            img, target = image.to(device), target.to(device)
            outputs = model(img)
            _, predicted = torch.max(outputs, 1)
            loss += nn.CrossEntropyLoss()(outputs, predicted).item()
            correctly_classified += (predicted == target).sum().item()
    loss /= len(test_loader.dataset)
    accuracy = 100.0 * correctly_classified / len(test_loader.dataset)

    return loss, accuracy




# trains model of client on client data
def train_client(model, device, optimizer, train_loader, epoch):
    for epochX in range(epoch):
        i=0
        for batch, (image, target) in enumerate(train_loader):
            img, target = image.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(img)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer.step()
            i+=1
    return loss.item()

def server_aggregate_geomedian(global_model, client_models,selected_clients):
    # get state of global model
    update_global_model = global_model.state_dict()
    # average the weights of selected clients and update global model
    points = [list(model.parameters()) for model in client_models]  # list of points, where each point is a list of tensors
    geoMedian = compute_geometric_median(points, weights = torch.ones(len(points)) ).median
    loop_counter = 0
    for weighti in update_global_model.keys():  # Pro Gewicht im Modell
        update_global_model[weighti] = geoMedian[loop_counter]  #Update das Gewicht mit dem mean (average) des Tensors
        global_model.load_state_dict(update_global_model)  # Update des Modells
        loop_counter = loop_counter + 1
    # update the models of all clients before next training
    for model in client_models:
        model.load_state_dict(global_model.state_dict())  # Update des Modells


# aggregates results of the selected and trained client models + takes the mean of the weights to update global model and all client models
def server_aggregate_mean(global_model, client_models,selected_clients):
    # get state of global model
    update_global_model = global_model.state_dict()
    # average the weights of selected clients and update global model
    for weighti in update_global_model.keys():  # Pro Gewicht im Modell
        update_global_model[weighti] = torch.stack( #Update das Gewicht mit dem mean (average) des Tensors
            [client_models[selected_clients[i]].state_dict()[weighti].float() for i in range(numberOfSelectedClients)], 0).mean(0)
        global_model.load_state_dict(update_global_model)  # Update des Modells
    # update the models of all clients before next training
    for model in client_models:
        model.load_state_dict(global_model.state_dict())  # Update des Modells


#Documentation
def getColumn(ws):
    pointer = 2
    for row in ws.iter_rows(2, 2, 2, 100):
        for cell in row:
            if cell.value is None:
                return pointer
            else:
                pointer += 1


def main():
    # Gathering Inputs
    print('Please enter a Number to select a Version:')
    print('Version 1: One worker is always wrong')
    print('Version 2: Same as 1, but with geometric median as solution')
    version = input("Input: ")
    print('\nIs This a documentation Run? (y / press enter):')
    doc = (True if input("Input: ").lower() == 'y' else False)
    if doc: # Legt die Anzahl der Runs fest
        runs = int(input('Please enter a number of runs: '))
        if runs <1 or runs > 100:
            print("invalid number of runs")
            return

    print('Starting...') #So i know wether the programm screwed up or not

    #Documentation
    if(doc):
        wb = load_workbook(path+'Version'+str(version)+'.xlsx')
    else:
        runs = 1

    for runIndex in range(runs):
        start = time.time()

        print("\n\nRun Number "+str(runIndex))
        print(":::::::::::::::::::::::::::::::::::::::::::::::::::::::")
        print(":::::::::::::::::::::::::::::::::::::::::::::::::::::::")
        if(doc):
            column = get_column_letter(getColumn(wb['Average loss during Training']))  # Nehme einfach das erst beste Sheet und gucke was die aktuelle Spalte/ was der aktuelle run ist

        torch.manual_seed(np.round(time.time() * 1000))

        # check if GPU is available, otherwise use CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

        # load MNIST dataset
        dataset_training = MNIST(root='./rootMNIST', train=True, download=True, transform=transform)
        #dataset_training2 = MNIST(root='./rootMNIST', train=True, download=True, transform=transform)

        dataset_testing = MNIST(root='./rootMNIST', train=False, download=True, transform=transform)

        # get the labels of each tuple (image, label) in the dataset
        labels = dataset_training.train_labels

        
        if(version != '1' and version != '2'):
            exit('Number should be 1, or 2')

        #set the dataset for one worker to the corrupt dataset
        #one corrupt worker has everything wrong and the others are untouched
        new_dataset_list = torch.utils.data.random_split(dataset_training,
                                                             [int(dataset_training.data.shape[0] / numberOfClients) for _ in
                                                              range(numberOfClients)])


        # for partX in partition gets 10x train_loader, one for each worker
        # train_loader is a list of DataLoaders, using new_dataset_list instead of partition_of_training_data loads the individually distributed datasets
        train_loader = [DataLoader(partX, batch_size=batchSize, shuffle=True) for partX in new_dataset_list]
        
        #actually set the corrupt worker
        dataset_corrupt = Dataset_RandomOther(transform)
        train_loader[0] = DataLoader(dataset_corrupt, batch_size=batchSize, shuffle=True)

        
        # further divide data of clients into train and test
        training_loader__local_client = []
        testing_loader__local_client = []
        for i in range(numberOfClients):
            clientDataset = new_dataset_list[i]
            #20 percent of data given to client is used for testing and the rest for training
            training_data_c, testing_data_c = train_test_split(clientDataset, test_size=0.2,shuffle=True,random_state=42)
            training_loader__local_client.append(DataLoader(training_data_c, batch_size=batchSize, shuffle=True))
            testing_loader__local_client.append(DataLoader(testing_data_c, batch_size=batchSize, shuffle=False))

        test_loader = DataLoader(dataset_testing, batch_size=batchSize, shuffle=False)

        print(" Number of Elements in Training Dataset:", len(dataset_training))
        print(" Number of Elements in Test Dataset:", len(dataset_testing))
        # initialize global server model and identical client models
        global_model = NeuralNetwork().to(device)
        client_models = [NeuralNetwork().to(device) for _ in range(numberOfClients)]
        print(" Model:", global_model)

        for model in client_models:
            model.load_state_dict(global_model.state_dict())
        # initialisation of Stochastic Gradient Descent as optimizer and  StepLR as scheduler for all client models
        optimizer = [torch.optim.AdamW(model.parameters(), learningrate) for model in client_models]
        scheduler = [StepLR(optim, step_size=1) for optim in optimizer]

        # waiting 0.1s so that prints can finish before tqdm
        time.sleep(0.1)

        for round in range(numberOfRounds):
            # to speed up training only train 4 randomly selected clients
            selectedClients = np.random.permutation(numberOfClients)[:numberOfSelectedClients]


            # train selected clients
            current_round_loss = 0
            for i in tqdm(range(numberOfSelectedClients), position=0, leave=True):
                current_round_loss += train_client(client_models[selectedClients[i]], device, optimizer[selectedClients[i]], training_loader__local_client[selectedClients[i]],
                                                   epoch=numberOfEpochs)

            print("Clients that have been trained:",selectedClients[0],",",selectedClients[1],", ",selectedClients[2],", ",selectedClients[3])

            # for each client test its model with test_loader of every client to get a 10x10 matrix
            tensorList = []
            clientid2 = 0 # documentation
            for clientid in range(numberOfClients):
                results = []
                for testi in (testing_loader__local_client):
                    test_loss, acc = test(client_models[clientid], device, testi)
                    results.append(acc)
                    # Documentation
                    if(doc):
                        wb['CM ' + str(clientid2) + str(clientid)][column + str(round + 2)].value = acc
                        clientid2 = (clientid2 + 1) % numberOfClients
                tensorList.append(results)
            torch.set_printoptions(linewidth=200)
            tensorList = torch.tensor(tensorList)
            print("Local Accuracy Matrix :")
            print(tensorList)

            # aggregate results of client training and update global- and all client models
            if version == '1':
                server_aggregate_mean(global_model, client_models, selectedClients)
            elif version == '2':
                server_aggregate_geomedian(global_model, client_models, selectedClients)



            # test current state of updated global model
            test_loss, accuracy = test(global_model, device, test_loader)
            print('Round %d :' % (round + 1), 'Average loss during Training: %0.3g | Global Test loss: %0.3g | Global Accuracy: %0.3f' % (
            current_round_loss / numberOfSelectedClients, test_loss, accuracy))
            # Documentation
            if(doc):
                wb['Average loss during Training'][column+str(round+2)].value = current_round_loss / numberOfSelectedClients
                wb['Global Test loss'][column + str(round + 2)].value = test_loss
                wb['Global Accuracy'][column + str(round + 2)].value = accuracy

            # adapt learning rates with scheduler
            for i in range(len(scheduler)):
                if i in selectedClients:
                    scheduler[i].step()

            # wait 0.1s so that prints can finish
            time.sleep(0.1)

            #Documentation
            if(doc):
                wb.save(path+'Version'+str(version)+'.xlsx')

        # Time orientation
        print("________________________")
        print("Time spent on this run: "+str((time.time()-start)/60)+"min")


if __name__ == '__main__':
    main()
