# Pytorch Machine Learning Worflow Exercise 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import math

# weight and bias initialization
WEIGHT = 4.56
BIAS = 14.3  
TOTAL_DATA = 1000

# create data set   
x = torch.randn(TOTAL_DATA, 1)
# y = (x * WEIGHT) + BIAS
y = (x * WEIGHT) + x/3 * 2*x + BIAS   

# move x and y data to GPU
x = x.cuda()
y = y.cuda()

# split data to 80%  training and 20% testing
train_size = int(0.8 * TOTAL_DATA)
train_x = x[:train_size]
train_y = y[:train_size]
test_x = x[train_size:]
test_y = y[train_size:]

# create model
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1) # input and output is 1 dimension

    def forward(self, x):
        return self.linear(x)

# instantiate model
model = LinearRegression()

# move model to GPU
model = model.cuda()

# create loss function
criterion = nn.MSELoss()

# create optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

# train model
epochs = 250
losses = []

model.train()
for i in range(epochs):
    i += 1
    # predict
    pred_y = model(train_x)
    # calculate loss
    loss = criterion(pred_y, train_y)
    # record loss
    losses.append(loss.item())
    # reset gradients
    optimizer.zero_grad()
    # backpropagate
    loss.backward()
    # update weights and bias
    optimizer.step()
    print(f'epoch: {i:2}  loss: {loss.item():10.8f}')       
    
# test model
with torch.no_grad():
    pred_y = model(test_x)
    loss = criterion(pred_y, test_y)
print(f'Testing Loss: {loss:.8f}')

# naje prediction
pred_y = model(test_x).detach().cpu().numpy()
# print(pred_y)
original_y = test_y.detach().cpu().numpy()

# print predicted weight and bias
print(f'\nPred weight: {model.linear.weight.item():.8f}, Pred bias: {model.linear.bias.item():.8f}')

# print actual weight and bias
print(f'Act weight: {WEIGHT:.8f}, Act bias: {BIAS:.8f}')

# print accuracy
print(f'Accuracy: {100 - loss:.4f}%')

# state_dict
print("\nModel's state_dict:")
print(model.state_dict())

# plot loss
plt.plot(range(epochs), losses)
plt.ylabel('Loss')
plt.xlabel('epoch');
plt.show()

# save model to pth file in /models subdirectory    
torch.save(model.state_dict(), './models/linear_regression.pth')



