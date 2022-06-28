from typing import Tuple, Dict

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import Tensor


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random

class Net(nn.Module):
    
    def __init__(self):
        
        super(Net, self).__init__()
        self.input_layer   = nn.Linear(9, 128) 
        self.hidden_layer1 = nn.Linear(128, 64) 
        self.hidden_layer2 = nn.Linear(64, 32) 
        self.hidden_layer3 = nn.Linear(32, 6) 
    
    def forward(self, x: Tensor):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer1(x))
        x = F.relu(self.hidden_layer2(x))
        
        return F.softmax(self.hidden_layer3(x), 1)


def load_data(file):
    
    dataset    = MotionSenseDataset()
    train_size = int(len(dataset) * 0.9)
    test_size  = len(dataset) - train_size
    
    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    
    train_loader  = DataLoader(train_set, batch_size=int(os.environ['BATCH_SIZE']), shuffle=True)
    test_loader   = DataLoader(test_set, batch_size=int(os.environ['BATCH_SIZE']), shuffle=False)
    
    num_samples = {'trainset' : len(train_set), 'testset' : len(test_set)}
    
    return train_loader, test_loader, num_samples



class MotionSenseDataset(Dataset):
    
    def __init__(self):
        dataset  = pd.read_csv('data/MotionSense.csv')

        if int(os.environ['NON_IID']) == 1:
        	user    = int(os.environ['USER_ID'])
        	dataset = dataset[dataset['subject'] == user]

        activity = dataset['activity'].values
        activity = LabelEncoder().fit_transform(activity)
        self.Y   = torch.as_tensor(activity)
        self.Y = F.one_hot(self.Y.long(), 6)
        dataset.drop('activity', axis=1, inplace=True)
        X        =  dataset.values.astype(np.float32)
        self.X = torch.from_numpy(X)
        self.n_samples = dataset.shape[0]
    
    def __getitem__(self, index):
        return self.X[index], self.Y[index]
        
    def __len__(self):
        return self.n_samples

def train(net, train_loader, epochs, device):
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    print(f"Training {epochs} epoch(s) w/ {len(train_loader)} batches each")
    
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        
        for i, data in enumerate(train_loader, 0):
            input_seq, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(input_seq)
            #print(outputs)
            loss = criterion(outputs, labels.float())
            
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # print every 100 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0


def test(
    net: Net,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
):
    """Validate the network on the entire test set."""
    criterion = nn.CrossEntropyLoss()
    correct   = 0
    total     = 0
    loss      = 0.0
    
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            
            loss += criterion(outputs, labels.float()).item()
            _, predicted = torch.max(outputs.data, 1)
            total   += labels.size(0)
            _, y = torch.max(labels, 1)
            correct += (predicted == y).sum().item()
    accuracy = correct / total
    return loss, accuracy

def main():
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Centralized PyTorch training")
    print("Load data")
    trainloader, testloader, _ = load_data()
    print("Start training")
    net=Net().to(DEVICE)
    train(net=net, trainloader=trainloader, epochs=2, device=DEVICE)
    print("Evaluate model")
    loss, accuracy = test(net=net, testloader=testloader, device=DEVICE)
    print("Loss: ", loss)
    print("Accuracy: ", accuracy)


if __name__ == "__main__":
    main()