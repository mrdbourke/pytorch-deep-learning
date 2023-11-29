# Computer Vision using Pytorch. using the fashion MNIST dataset,
# download the train and test dataset and create a dataloader for each of them.
# visualize 10 images from the train dataset using matplotlib.

import torch
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import time
from tqdm import tqdm
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader



EPOCHS = 50         # 50x: 97% Test Accuracy
                    # 100x:98% Test Accuracy 
                    # 200x:99% Test Accuracy  
                    
# 60,000 images in the train dataset                     
BATCH_SIZE = 128    # 16 -> 602 sec. 
                    # 32 -> 455 sec.
                    # 64 -> 370 sec. -> 173 sec. pin_memory=True, workers=4 **BEST**
                    # 64              > 395 sec. pin_memory=True, workers=0
                    # 64              > 195 sec. pin_memory=False, workers=4
                    # 64              > 201 sec. pin_memory=False, workers=8
                    # 128 -> 336 sec.-> 153 sec. pin_memory=True, workers=4 **BEST**
                    # 256 -> 316 sec.
                    # 512 -> 307 sec.


# Download the train and test dataset and create a dataloader for each of them.
# download into the data folder
train_data = datasets.FashionMNIST(root='./data', 
                                   train=True, 
                                   download=True, 
                                   transform=transforms.ToTensor())

test_data = datasets.FashionMNIST(root='./data', 
                                  train=False, 
                                  download=True, 
                                  transform=transforms.ToTensor())   

# print the shape of the train and test data
# print(train_data.data.shape)
# print(test_data.data.shape)

# print the labels of the train and test data
# print(train_data.targets)
# print(test_data.targets)

# plot 10 images from the train dataset using matplotlib.
# define the labels of the dataset
# labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
#           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# print('data targets:', train_data.targets[0:10])
# print('data labels:', train_data.class_to_idx)

# define the labels of the dataset
labels = list(train_data.class_to_idx.keys())
# print('labels:', labels)

# plot the images
fig = plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(train_data.data[i], cmap='gray')
    plt.title(labels[train_data.targets[i]])
    plt.axis('off')

# Increase padding
plt.subplots_adjust(wspace=0.95, hspace=0.95)
plt.show()

# use dataloader to load the data

# create a dataloader for the train data
train_loader = torch.utils.data.DataLoader(train_data, 
                                           batch_size=BATCH_SIZE, 
                                           shuffle=True,
                                           num_workers=4,
                                           pin_memory=True)

# create a dataloader for the test data
test_loader = torch.utils.data.DataLoader(test_data, 
                                          batch_size=BATCH_SIZE, 
                                          shuffle=True)

# print the shape of the train and test loader
# print(train_loader.dataset.data.shape)
# print(test_loader.dataset.data.shape)

# print total size and bathces of the train and test loader
# print(len(train_loader.dataset))
# print(len(train_loader))

# print the shape of the first batch of the train loader
dataiter = iter(train_loader)
images, labels = next(dataiter)
# print(images.shape)
# print(labels.shape)

# create a model
# define the model
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(784, 512) # 28px X 29px = 784
        self.linear2 = torch.nn.Linear(512, 256)
        self.linear3 = torch.nn.Linear(256, 128)
        self.linear4 = torch.nn.Linear(128, 64)
        self.linear5 = torch.nn.Linear(64, 10)
        
    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.nn.functional.relu(self.linear1(x))
        x = torch.nn.functional.relu(self.linear2(x))
        x = torch.nn.functional.relu(self.linear3(x))
        x = torch.nn.functional.relu(self.linear4(x))
        x = self.linear5(x)
        return x
    
# create a model
model = Model()

# define the loss function
criterion = torch.nn.CrossEntropyLoss()

# define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# use gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"



# define the lists to store the loss and accuracy
train_loss = []
train_accuracy = []
test_loss = []
test_accuracy = []

def training_loop(model: Module, epochs: int, train_loader: DataLoader, loss: Module, optimizer: Optimizer, device: str):
    model.to(device)
    for epoch in tqdm(range(epochs)):
        # train the model
        correct = 0
        for i, (images, labels) in enumerate(train_loader):
            # move data to gpu
            images = images.to(device)
            labels = labels.to(device)
            
            # make the predictions
            outputs = model(images)
            
            # calculate the loss
            loss = criterion(outputs, labels)
            
            # calculate gradients
            loss.backward()
            
            # update parameters
            optimizer.step()
            
            # clear gradients
            optimizer.zero_grad()
            
            # get the predictions
            _, predicted = torch.max(outputs.data, 1)
            
            # update the number of correct predictions
            correct += (predicted == labels).sum().item()          
            
            # print the loss and accuracy for every 100th batch
            if (i+1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch+1, EPOCHS, i+1, len(train_loader), loss.item()))
            
        # update the train loss and accuracy for the epoch
        train_loss.append(loss.item())
        train_accuracy.append(100 * correct / len(train_loader.dataset))
        
        # print the loss and accuracy for the epoch
        # print('Epoch [{}/{}], Train Loss: {:.4f}, Train Accuracy: {:.2f}%'
        #     .format(epoch+1, EPOCHS, train_loss[-1], train_accuracy[-1]))
    
        # test the model
        correct = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(test_loader):
                # move data to gpu 
                images = images.to(device)
                labels = labels.to(device)
                
                # make the predictions
                outputs = model(images)
                
                # calculate the loss
                loss = criterion(outputs, labels)
                
                # get the predictions
                _, predicted = torch.max(outputs.data, 1)
                
                # update the number of correct predictions
                correct += (predicted == labels).sum().item()
                
        # update the test loss and accuracy for the epoch
        test_loss.append(loss.item())
        test_accuracy.append(100 * correct / len(test_loader.dataset))
        
        # print the loss and accuracy for the epoch
        print('Epoch [{}/{}], Train Loss: {:.4f}, Train Accuracy: {:.2f}%, Test Loss: {:.4f}, Test Accuracy: {:.2f}%'
                .format(epoch+1, EPOCHS, train_loss[-1], train_accuracy[-1], test_loss[-1], test_accuracy[-1]))
    
  


# call the training loop
start_time = time.time()
training_loop(model, EPOCHS, train_loader, criterion, optimizer, device)
# print elapsed time
print('Elapsed time: {:.2f} seconds'.format(time.time() - start_time))



# train the model
# start_time = time.time()
# for epoch in tqdm(range(EPOCHS)):
#     # train the model
#     correct = 0
#     for i, (images, labels) in enumerate(train_loader):
#         # move data to gpu
#         images = images.to(device)
#         labels = labels.to(device)
        
#         # make the predictions
#         outputs = model(images)
        
#         # calculate the loss
#         loss = criterion(outputs, labels)
        
#         # calculate gradients
#         loss.backward()
        
#         # update parameters
#         optimizer.step()
        
#         # clear gradients
#         optimizer.zero_grad()
        
#         # get the predictions
#         _, predicted = torch.max(outputs.data, 1)
        
#         # update the number of correct predictions
#         correct += (predicted == labels).sum().item()
        
#         # print the loss and accuracy for every 100th batch
#         if (i+1) % 100 == 0:
#             print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
#                   .format(epoch+1, EPOCHS, i+1, len(train_loader), loss.item()))
    
#     # update the train loss and accuracy for the epoch
#     train_loss.append(loss.item())
#     train_accuracy.append(100 * correct / len(train_loader.dataset))
    
#     # test the model
#     correct = 0
#     with torch.no_grad():
#         for i, (images, labels) in enumerate(test_loader):
#             # move data to gpu 
#             images = images.to(device)
#             labels = labels.to(device)
            
#             # make the predictions
#             outputs = model(images)
            
#             # calculate the loss
#             loss = criterion(outputs, labels)
            
#             # get the predictions
#             _, predicted = torch.max(outputs.data, 1)
            
#             # update the number of correct predictions
#             correct += (predicted == labels).sum().item()
            
#     # update the test loss and accuracy for the epoch
#     test_loss.append(loss.item())
#     test_accuracy.append(100 * correct / len(test_loader.dataset))
    
#     # print the loss and accuracy for the epoch
#     print('Epoch [{}/{}], Train Loss: {:.4f}, Train Accuracy: {:.2f}%, Test Loss: {:.4f}, Test Accuracy: {:.2f}%'
#           .format(epoch+1, EPOCHS, train_loss[-1], train_accuracy[-1], test_loss[-1], test_accuracy[-1]))
 


    
    











 