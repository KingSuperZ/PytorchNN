""" Finds the classification of data points using PyTorch

It takes a 1D tensor and a 2D tensor with the 2D bring the features and the 1D
being the targets and trains the algorithm through gradient descent to eventually
test the data on itself.
"""

import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from sklearn.datasets import make_classification

## Data ##
#X = torch.tensor([[1,2],[1,4],[2,3],[2,1],[3,1],[2,2],[6,5],[7,8],[8,7],[8,6],[9,10],[11,10]], dtype = torch.float32) # Features
#y = torch.tensor([0,0,0,0,0,0,1,1,1,1,1,1], dtype = torch.float32) # Targets
X, y = make_classification(n_samples = 100, n_features = 2, n_redundant = 0, n_informative = 2, n_clusters_per_class = 1, flip_y = 0, random_state = 61)
X = torch.tensor(X, dtype = torch.float32)
y = torch.tensor(y, dtype = torch.float32)
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size = 0.66)

## Model ##
# model = nn.Sequential(nn.Linear(2,1), nn.Sigmoid(), nn.Linear(1,1), nn.Sigmoid())
# the code below replaces the code above
class mynn(nn.Module): ## nn.Module is the parent
  # lin is a class variable that is always public and it is a linear layer
  # sig is a class variable that also public and is the Sigmoid
  def __init__(self): # The constructor of the class
    """in nn.Linear the first number is the previous layer and second number is the current layer
    in this model the input layer has two nodes with one hidden layer with three nodes and an output layer with one node"""
    super().__init__() # super() is the parent and is nn.Module and super().__init__() is constructor of the module
    # Builds the neural network
    self.lin = nn.Linear(2,12) # The first number represents the input layer and the second represents the current hidden layer
    self.sig = nn.Sigmoid() # nn.Sigmoid sets any number to a number between 0 and 1 using the equation y = 1/(1+e^-x)
    self.lin2 = nn.Linear(12,1) # The first number is the same as the previous number and the second number represents the current output layer
    self.sig2 = nn.Sigmoid()

  def forward(self, x): # The x in this line represents the input of the NN
    x = self.lin(x)
    x = self.sig(x)
    x = self.lin2(x)
    x = self.sig2(x)
    return x # The x in this line represents the output after going through the entire NN
model = mynn()

## Training/Optimization ##
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) # the GD in SGD stands for gradient descent
n_epochs = 100
lossList = []
for i in tqdm(range(n_epochs)): # tqdm simply adds a loading bar with the eta for when the loop completes
  for i in range(len(Xtrain)):
    pred = model(Xtrain[i]) # Holds the prediction in a variable. If model() is called then it runs the forward function
    loss = loss_fn(pred[0], ytrain[i]) # Holds the loss in a variable
    loss.backward() # Computes the derivative/slope of the error
    optimizer.step() # Takes the step towards the local min using the lr
    optimizer.zero_grad() # Resets the slope calculation in order for change to occur
  lossList.append(loss.detach())
print(lossList)
plt.plot(lossList) # The plot should mainly decrease but it sometimes might increase due to lack of data or insufficient epochs

## Testing ##
ypred = model(Xtest).round() # Holds the prediction in a variable

## Analysis ##
print(ypred)
print(ytest)
#ypred = torch.squeeze(ypred, axis = 1)
#scoreDiff = ypred-ytest
#accuracyScore = (len(scoreDiff)-scoreDiff.sum())/len(scoreDiff)
#print(accuracyScore)
accuracyScore = accuracy_score(ytest.detach().numpy(), ypred.detach().numpy())
print(accuracyScore * 100)
plt.figure()
plt.scatter(Xtrain[:,0],Xtrain[:,1], c = ytrain, cmap = "cool")
plt.scatter(Xtest[:,0],Xtest[:,1], c = ypred.detach().numpy(), marker = "x", s = 50, cmap = "cool")
plt.axis("equal")
plt.grid()
