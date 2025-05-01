
#Imports

import torch


"""
    @create_model is a class meant for creating a simple 6-layer neural network . It can also return a single layer based on the function call

    functions: 
    init: takes two arguments which are used for creating single neuron 
    
    show_model: prints the model tensor along with the parameters
    
    sdg: a simple sigmoid gradient descent implementation. torch.optim.SGD can also be used in place for computing that
    
    torch.optim.SGD(model.paramters(), lr = 1e-2, momentum = 0.8)  
        where lr is learning rate, and momentum is for the pace at which the optimizer is supposed to calcualte the minima

    generate_neural_network: generates a 6-layer neural network with same padding, stride and kernel size

"""
class create_model:

    def __init__(self, in_features: int = 1, out_features: int = 1):
        self.toy_model = torch.nn.Linear(in_features, out_features)

    def show_model(self):
        print("Here is your model:", self.toy_model)        

    '''
    Simple function to computer sigmoid gradient descent. It can also be done via calling SGD from torch.optim 
    This funtion works in the following steps:
    1. initialize a tensor of batch size equal to the out_features and dimensions of the in_features of the toy model
    2. The loop iterates through the limit and performs the following operations:
        - passes the model xb tensor
        - calculates mean squared error
        - appends to a list
        - zero_grad() methos to set the gradients of the model as zeroes before backpropagation as backprop adds the gradients with the previous gradients
        - computes the SGD by subtracting the parameters with grad * learning rate

    '''
    def sgd(self, limit: int = 1000):
        self.log = []
        self.ref_tensor = torch.tensor([[1.0, 1.0]])
        self.xb = torch.randn(self.toy_model.out_features, self.toy_model.in_features)
        for x in range(limit):
            self.y_b = self.toy_model(self.xb)
            self.loss = ((self.y_b - self.ref_tensor)**2).sum(1).mean()
            self.log.append(self.loss.item())
            self.toy_model.zero_grad()   #initialize gradients to zero before backpropagation
            self.loss.backward()
            with torch.no_grad():
                for prm in self.toy_model.parameters():
                    prm[...] -= 0.01 * prm.grad  
        return prm

    '''
        Generates a Neural Network with ReLU (Rectified Linear Unit) as activation function and a 2D convolution layer with the provided arguments
    '''
    def generate_neural_network(self):
        self.neural_network = torch.nn.Sequential(torch.nn.Conv2d(1, 4, kernel_size=2, padding= 2, stride=2), 
                                     torch.nn.ReLU(),
                                     torch.nn.Conv2d(in_channels=4, out_channels=8, kernel_size=2, padding= 2, stride=2), 
                                     torch.nn.ReLU(),
                                     torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=2, padding= 2, stride=2), 
                                     torch.nn.ReLU(),
                                     torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, padding= 2, stride=2), 
                                     torch.nn.ReLU(),
                                     torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, padding= 2, stride=2), 
                                     torch.nn.ReLU(),
                                     torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, padding= 2, stride=2), 
                                     torch.nn.ReLU(),
                                     torch.nn.AdaptiveAvgPool2d(1),
                                     torch.nn.Flatten(),
                                     torch.nn.Linear(64,8))
        return self.neural_network

    def display_neural_network(self):
        print ("Neural Network\n", self.neural_network)



if(__name__ == "__main__"):

    model = create_model(3,2)

    model.show_model()

    print("Tensor after SGD",model.sgd())

    model.generate_neural_network()

    model.display_neural_network()