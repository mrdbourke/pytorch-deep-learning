from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


# print("Loading data...")

x, y = make_circles(n_samples=1000, noise=0.1, factor=0.2, random_state=1)

print(x.shape)
# print(x.dtype)
# print(len(x))
# print(len(y))
print(x[:10])
print(y[:10])

x = x.astype(np.float32)
y = y.astype(np.float32)
print(x.dtype) 
print(y.dtype)
print(x[:10])
print(y[:10])



plt.scatter(x[:, 0], x[:, 1], c=y, s=30, cmap="RdBu")
plt.title('Data')
# plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

 
