# Create a multiclass classification using sklearn make_blobs
# and LogisticRegression and pytorch nn.Linear and nn.CrossEntropyLoss
# and visualise the results

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
import time

# Create a color map for the scatter plot
cmap = ListedColormap(['red', 'green', 'blue', 'orange', 'purple'])

# Create a dataset 
N_SAMPLES = 3000
N_FEATURES = 2
N_CLASSES = 6
N_NEURONS = 8
TEST_SIZE = 0.1
CLUSTER_STD = 1.3
epochs = 200 
lr = 0.01

# Create a dataset with 1000 samples, 2 features and 4 classes
X, y = make_blobs(n_samples=N_SAMPLES, n_features=N_FEATURES, centers=N_CLASSES, cluster_std=CLUSTER_STD, random_state=42)

# print(X.shape)
# print(y.shape)
# print(X[:15])
# print(y[:15])

# convert X, y to tensors
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).long().unsqueeze(1)

# print(X.shape)
# print(y.shape)

# Visualise the dataset using matplotlib scatter
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap)
plt.show()

# split dataset into train and test using sklearn train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)

# print(X_train.shape)
# print(y_train.shape)
# print(X_test.shape)
# print(y_test.shape)

# move tensors to cuda if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # use gpu if available
device = torch.device("cpu")   # use cpu

X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

# convert y_train, y_test to one-hot encoding   
y_train = y_train.to(torch.float32)
y_test = y_test.to(torch.float32)

# print(X_train.dtype)
# print(y_train.dtype)
# print(X_test.dtype)
# print(y_test.dtype)

# create a logistic regression model using nn.Sequential, with 2 hidden layers
# 2 input features, 4 output classes    
model = nn.Sequential(
    nn.Linear(N_FEATURES, N_NEURONS),
    # nn.ReLU(),
    nn.Linear(N_NEURONS, N_NEURONS),
    # nn.ReLU(),
    nn.Linear(N_NEURONS, N_CLASSES),
    # nn.ReLU()
)

# print(model)

# move model to cuda if available
model.to(device)

# define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr)

# train the model, and save the losses per epoch in a list and plot it
losses = []

start_time = time.time()
model.train()
for epoch in range(epochs):
    # forward pass
    y_pred = model(X_train)
    # calculate loss
    loss = criterion(y_pred, y_train.squeeze(1).long())
    # backward pass
    loss.backward()
    # update parameters
    optimizer.step()
    # clear gradients
    optimizer.zero_grad()
    # save loss
    losses.append(loss.item())
    # print loss
    if (epoch+1) % 10 == 0:
        print(f'Epoch: {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

end_time = time.time()

# plot losses   
plt.plot(losses)
plt.title('Losses')
plt.show()

# evaluate the model on test set
model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    # calculate loss
    loss = criterion(y_pred, y_test.squeeze(1).long())
    print(f'Test Loss: {loss.item():.4f}')
    
# compare the predictions with the ground truth and give as percentage of correct predictions
_, predicted = torch.max(y_pred, 1)
correct = (predicted == y_test.squeeze(1)).sum().item()
accuracy = correct/len(y_test)*100
print(f'\nCorrect Predictions: {correct}/{len(y_test)}, {correct/len(y_test)*100:.2f}%')

# Calculate and print the elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")

# Visualise the predictions using matplotlib scatter using different colors for each class
# move data to cpu for plotting and show segmentation
# move data to cpu for plotting and show segmentation


X_test = X_test.to('cpu')
y_test = y_test.to('cpu')
predicted = predicted.to('cpu')

plt.figure(figsize=(10,5))

# Plot predicted labels
plt.subplot(1, 2, 1)
plt.scatter(X_test[:, 0], X_test[:, 1], c=predicted, cmap=cmap)
plt.title('Predicted Labels (Accuracy: {:.2f}%)'.format(accuracy))

# Plot true labels
plt.subplot(1, 2, 2)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test.squeeze(1), cmap=cmap)
plt.title('True Labels')

plt.show()

if (N_FEATURES == 2):
    # plot decision boundary
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    h = .02  # step size in the mesh
    x_min, x_max = X_test[:, 0].min() - .5, X_test[:, 0].max() + .5
    y_min, y_max = X_test[:, 1].min() - .5, X_test[:, 1].max() + .5
    xx, yy = torch.meshgrid(torch.arange(x_min, x_max, h),
                         torch.arange(y_min, y_max, h))
    # move to gpu device for prediction'
    xx = xx.to(device)
    yy = yy.to(device)


    Z = model(torch.cat((xx.reshape(-1,1), yy.reshape(-1,1)), dim=1))
    Z = torch.argmax(Z, dim=1)
    Z = Z.reshape(xx.shape) # Put the result into a color plot
    # move to cpu device for plotting
    xx = xx.to('cpu')
    yy = yy.to('cpu')
    Z = Z.to('cpu')

    plt.figure(figsize=(10,5))
    plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.5)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test.squeeze(1), cmap=cmap)
    plt.title('Decision Boundary')
    plt.show()
    
# # plot decision boundary
# # Plot the decision boundary. For that, we will assign a color to each
# # point in the mesh [x_min, x_max]x[y_min, y_max].
# h = .02  # step size in the mesh
# x_min, x_max = X_test[:, 0].min() - .5, X_test[:, 0].max() + .5
# y_min, y_max = X_test[:, 1].min() - .5, X_test[:, 1].max() + .5
# xx, yy = torch.meshgrid(torch.arange(x_min, x_max, h),
#                      torch.arange(y_min, y_max, h))
# # move to gpu device for prediction'
# xx = xx.to(device)
# yy = yy.to(device)


# Z = model(torch.cat((xx.reshape(-1,1), yy.reshape(-1,1)), dim=1))
# Z = torch.argmax(Z, dim=1)
# Z = Z.reshape(xx.shape) # Put the result into a color plot
# # move to cpu device for plotting
# xx = xx.to('cpu')
# yy = yy.to('cpu')
# Z = Z.to('cpu')

# plt.figure(figsize=(10,5))
# plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.5)
# plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test.squeeze(1), cmap=cmap)
# plt.title('Decision Boundary')
# plt.show()

