import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional
import torch.optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from torchvision.datasets import MNIST
from tqdm import tqdm
import skimage

# definition of Parameters
numberOfEpochs = 3
batchSize = 12
learningrate = 0.001
numberOfClients = 10
numberOfSelectedClients = 4
numberOfRounds = 20


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
            image, target = torch.tensor(skimage.util.random_noise(image, mode='s&p', salt_vs_pepper=0.5, clip=True)).to(device), target.to(device)
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            loss += nn.CrossEntropyLoss()(outputs, predicted).item()
            correctly_classified += (predicted == target).sum().item()

    loss /= len(test_loader.dataset)
    accuracy = 100.0 * correctly_classified / len(test_loader.dataset)

    return loss, accuracy




# trains model of client on client data
def train_client(model, device, optimizer, train_loader, epoch):
    for epochX in range(epoch):
        for batch, (image, target) in enumerate(train_loader):
            image, target = torch.tensor(skimage.util.random_noise(image, mode='s&p', salt_vs_pepper=0.5, clip=True)).to(device), target.to(device)
            optimizer.zero_grad()
            output = model(image)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer.step()
    return loss.item()


# aggregates results of the selected and trained client models + takes the mean of the weights to update global model and all client models
def server_aggregate(global_model, client_models,selected_clients):
    # get state of global model
    update_global_model = global_model.state_dict()
    # average the weights of selected clients and update global model
    for weighti in update_global_model.keys():
        update_global_model[weighti] = torch.stack(
            [client_models[selected_clients[i]].state_dict()[weighti].float() for i in range(numberOfSelectedClients)], 0).mean(0)
        global_model.load_state_dict(update_global_model)
    # update the models of all clients before next training
    for model in client_models:
        model.load_state_dict(global_model.state_dict())


def main():
    torch.manual_seed(np.round(time.time() * 1000))

    # check if GPU is available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    # load MNIST dataset
    dataset_training = MNIST(root='./rootMNIST', train=True, download=True, transform=transform)
    dataset_training2 = MNIST(root='./rootMNIST', train=True, download=True, transform=transform)

    dataset_testing = MNIST(root='./rootMNIST', train=False, download=True, transform=transform)

    # get the labels of each tuple (image, label) in the dataset
    labels = dataset_training.train_labels

    print('Please enter a Number to select a Version:')
    print('Version 1: One worker with mainly 0es')
    print('Version 2: Every worker with mainly one number')
    print('Version 3: Equal distribution of all numbers')

    version = input()
    if version == '1':
        # Version 1
        # one worker has the majority of one number but still a few pictures of other numbers

        # get the indices of all tuples, where the label is '0'
        label_zero_indices = (labels == 0).nonzero()

        # from tensor to list with flatten().tolist()
        label_zero_indices = label_zero_indices.flatten().tolist()
        # print(len(label_zero_indices)) = 5923

        # get a subset of the dataset with the filtered indices
        label_zero_subset = torch.utils.data.Subset(dataset_training, label_zero_indices)

        # random_split expects a dataset and the wanted length, returns 10 non-overlapping new datasets, meaning a list of datasets?
        partition_of_training_data = torch.utils.data.random_split(label_zero_subset,
                                                                   [5023, 100, 100, 100, 100, 100, 100, 100, 100,
                                                                    100])  # 5023 + 9*100 = 5923

        # make a subset of the mnist dataset without zeroes: indices of all tuples where the label is NOT '0'
        label_not_zero_indices = (labels != 0).nonzero()
        label_not_zero_indices = label_not_zero_indices.flatten().tolist()
        # print(len(label_not_zero_indices)) = 54077

        # make the second subset without zeroes
        label_not_zero_subset = torch.utils.data.Subset(dataset_training, label_not_zero_indices)
        partition_of_training_data2 = torch.utils.data.random_split(label_not_zero_subset,
                                                                    [977, 5900, 5900, 5900, 5900, 5900, 5900, 5900,
                                                                     5900, 5900])  # 977 + 9*5900 = 54077

        # use ConcatDataset to join a list of Datasets
        new_dataset_list = []
        for i in range(10):
            new_dataset_list.append(
                torch.utils.data.ConcatDataset([partition_of_training_data[i], partition_of_training_data2[i]]))

    elif version == '2':
        # get the indices of the numbers
        label_zero_indices = (labels == 0).nonzero()
        label_one_indices = (labels == 1).nonzero()
        label_two_indices = (labels == 2).nonzero()
        label_three_indices = (labels == 3).nonzero()
        label_four_indices = (labels == 4).nonzero()
        label_five_indices = (labels == 5).nonzero()
        label_six_indices = (labels == 6).nonzero()
        label_seven_indices = (labels == 7).nonzero()
        label_eight_indices = (labels == 8).nonzero()
        label_nine_indices = (labels == 9).nonzero()

        # from tensor to list
        label_zero_indices = label_zero_indices.flatten().tolist()
        label_one_indices = label_one_indices.flatten().tolist()
        label_two_indices = label_two_indices.flatten().tolist()
        label_three_indices = label_three_indices.flatten().tolist()
        label_four_indices = label_four_indices.flatten().tolist()
        label_five_indices = label_five_indices.flatten().tolist()
        label_six_indices = label_six_indices.flatten().tolist()
        label_seven_indices = label_seven_indices.flatten().tolist()
        label_eight_indices = label_eight_indices.flatten().tolist()
        label_nine_indices = label_nine_indices.flatten().tolist()

        # make subsets for each number
        label_zero_subset = torch.utils.data.Subset(dataset_training, label_zero_indices)
        label_one_subset = torch.utils.data.Subset(dataset_training, label_one_indices)
        label_two_subset = torch.utils.data.Subset(dataset_training, label_two_indices)
        label_three_subset = torch.utils.data.Subset(dataset_training, label_three_indices)
        label_four_subset = torch.utils.data.Subset(dataset_training, label_four_indices)
        label_five_subset = torch.utils.data.Subset(dataset_training, label_five_indices)
        label_six_subset = torch.utils.data.Subset(dataset_training, label_six_indices)
        label_seven_subset = torch.utils.data.Subset(dataset_training, label_seven_indices)
        label_eight_subset = torch.utils.data.Subset(dataset_training, label_eight_indices)
        label_nine_subset = torch.utils.data.Subset(dataset_training, label_nine_indices)

        # get the count for each number
        # print(len(label_zero_indices)) = 5923
        # print(len(label_one_indices)) = 6742
        # print(len(label_two_indices)) = 5958
        # print(len(label_three_indices)) = 6131
        # print(len(label_four_indices)) = 5842
        # print(len(label_five_indices)) = 5421
        # print(len(label_six_indices)) = 5918
        # print(len(label_seven_indices)) = 6265
        # print(len(label_eight_indices)) = 5851
        # print(len(label_nine_indices)) = 5949

        # split the numbers for the workers
        partition_of_training_data0 = torch.utils.data.random_split(label_zero_subset,
                                                                    [5023, 100, 100, 100, 100, 100, 100, 100, 100, 100])
        partition_of_training_data1 = torch.utils.data.random_split(label_one_subset,
                                                                    [100, 5842, 100, 100, 100, 100, 100, 100, 100, 100])
        partition_of_training_data2 = torch.utils.data.random_split(label_two_subset,
                                                                    [100, 100, 5058, 100, 100, 100, 100, 100, 100, 100])
        partition_of_training_data3 = torch.utils.data.random_split(label_three_subset,
                                                                    [100, 100, 100, 5231, 100, 100, 100, 100, 100, 100])
        partition_of_training_data4 = torch.utils.data.random_split(label_four_subset,
                                                                    [100, 100, 100, 100, 4942, 100, 100, 100, 100, 100])
        partition_of_training_data5 = torch.utils.data.random_split(label_five_subset,
                                                                    [100, 100, 100, 100, 100, 4521, 100, 100, 100, 100])
        partition_of_training_data6 = torch.utils.data.random_split(label_six_subset,
                                                                    [100, 100, 100, 100, 100, 100, 5018, 100, 100, 100])
        partition_of_training_data7 = torch.utils.data.random_split(label_seven_subset,
                                                                    [100, 100, 100, 100, 100, 100, 100, 5365, 100, 100])
        partition_of_training_data8 = torch.utils.data.random_split(label_eight_subset,
                                                                    [100, 100, 100, 100, 100, 100, 100, 100, 4951, 100])
        partition_of_training_data9 = torch.utils.data.random_split(label_nine_subset,
                                                                    [100, 100, 100, 100, 100, 100, 100, 100, 100, 5049])

        # concat datasets
        new_dataset_list = []
        for i in range(10):
            new_dataset_list.append(torch.utils.data.ConcatDataset(
                [partition_of_training_data0[i], partition_of_training_data1[i], partition_of_training_data2[i],
                 partition_of_training_data3[i], partition_of_training_data4[i], partition_of_training_data5[i],
                 partition_of_training_data6[i], partition_of_training_data7[i], partition_of_training_data8[i],
                 partition_of_training_data9[i]]))

    elif version == '3':
        # Dividing the training data into num_clients, with each client having equal number of images
        new_dataset_list = torch.utils.data.random_split(dataset_training,
                                                         [int(dataset_training.data.shape[0] / numberOfClients) for _ in
                                                          range(numberOfClients)])

    else:
        exit('Number should be 1, 2 or 3')

    # for partX in partition gets 10x train_loader, one for each worker
    # train_loader is a list of DataLoaders, using new_dataset_list instead of partition_of_training_data loads the individually distributed datasets
    train_loader = [DataLoader(partX, batch_size=batchSize, shuffle=True) for partX in new_dataset_list]

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
        for clientid in range(numberOfClients):
            results = []
            for testi in (testing_loader__local_client):
                test_loss, acc = test(client_models[clientid], device, testi)
                results.append(acc)
            tensorList.append(results)
        torch.set_printoptions(linewidth=200)
        tensorList = torch.tensor(tensorList)
        print("Local Accuracy Matrix :")
        print(tensorList)

        # aggregate results of client training and update global- and all client models
        server_aggregate(global_model, client_models,selectedClients)



        # test current state of updated global model
        test_loss, accuracy = test(global_model, device, test_loader)
        print('Round %d :' % (round + 1), 'Average loss during Training: %0.3g | Global Test loss: %0.3g | Global Accuracy: %0.3f' % (
        current_round_loss / numberOfSelectedClients, test_loss, accuracy))

        # adapt learning rates with scheduler
        for i in range(len(scheduler)):
            if i in selectedClients:
                scheduler[i].step()

        # wait 0.1s so that prints can finish
        time.sleep(0.1)


if __name__ == '__main__':
    main()
