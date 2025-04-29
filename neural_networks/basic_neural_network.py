#This code is a work in progress. Will be pushing the updated code 

#Imports

import torch
from sklearn import model_selection

#creating a simple model with the help of torch.nn module

toy_model = torch.nn.Linear(4,2)  #here number of input features = 4 and number of output features = 2

toy_model(torch.randn(4,4))  #passing random inputs to the model