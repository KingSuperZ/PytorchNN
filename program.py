""" Finds the classification of data points using PyTorch

It takes a 1D tensor and a 2D tensor with the 2D bring the features and the 1D
being the targets and trains the algorithm through gradient descent to eventually
test the data on itself
"""

import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Data
X = torch.tensor([[1.0,2],[1,4],[2,3],[2,1],[3,1],[6,5],[7,8],[8,7],[8,6],[9,10]]) # Features
y = torch.tensor([0.0,0,0,0,0,1,1,1,1,1]) # Targets
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)

# Model
# model = nn.Sequential(nn.Linear(2,1), nn.Sigmoid(), nn.Linear(1,1), nn.Sigmoid())
# the code below replaces the code above
class mynn(nn.Module): ## nn.Module is the parent
  def __init__(self):
    super().__init__() # super() is the parent and is nn.Module and super().__init__() is constructor of the module
    self.lin = nn.Linear(2,1) # lin is a class variable that is always public and it is a linear layer
    self.sig = nn.Sigmoid() # sig is a class variable that also public and is the Sigmoid
    self.lin2 = nn.Linear(1,1)
    self.sig2 = nn.Sigmoid()
  def forward(self, x):
    x = self.lin(x)
    x = self.sig(x)
    x = self.lin2(x)
    x = self.sig2(x)
    return x
model = mynn()
# Training/Optimization
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3) # the GD in SGD stands for gradient descent
n_epochs = 500
lossList = []
for i in range(n_epochs):
  for i in range(len(Xtrain)):
    pred = model(Xtrain[i]) # Holds the prediction in a variable
    loss = loss_fn(pred[0], ytrain[i]) # Holds the loss in a variable
    loss.backward() # Computes the derivative/slope of the error
    optimizer.step() # Takes the step towards the local min using the lr
    optimizer.zero_grad() # Resets the slope calculation in order for change to occur
  lossList.append(loss.detach())
print(lossList)
plt.plot(lossList) # The plot should mainly decrease but it sometimes might increase due to lack of data or insufficient epochs

# Testing
ypred = model(Xtest).round() # Holds the prediction in a variable

# Analysis
print(ypred)
print(ytest)
ypred = torch.squeeze(ypred, axis = 1)
scoreDiff = ypred-ytest
accuracyScore = (len(scoreDiff)-scoreDiff.sum())/len(scoreDiff)
print(accuracyScore)
