
"""
    @brief Imports
    These are the imports used throughout the code
    Libraries used: Pytorch, Scikit-Learn
"""
import torch
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split 
from torch.utils.data import DataLoader, Dataset 
import torch.nn.functional as F 

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
        print("Here is your model:\n", self.toy_model)        

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
    def sgd(self, limit: int = 1000, lr = (3*1e-2)):
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
                    prm[...] -= lr * prm.grad  
        return prm

    '''
        Generates a Neural Network with ReLU (Rectified Linear Unit) as activation function and a 2D convolution layer with the provided arguments
    '''
    def generate_neural_network(self):
        self.neural_network = torch.nn.Sequential(torch.nn.Conv2d(1, 4, kernel_size=3, padding= 1, stride=2), 
                                     torch.nn.ReLU(),
                                     torch.nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding= 1, stride=2), 
                                     torch.nn.ReLU(),
                                     torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding= 1, stride=2), 
                                     torch.nn.ReLU(),
                                     torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding= 1, stride=2), 
                                     torch.nn.ReLU(),
                                     torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding= 1, stride=2), 
                                     torch.nn.ReLU(),
                                     torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding= 1, stride=2), 
                                     torch.nn.ReLU(),
                                     torch.nn.AdaptiveAvgPool2d(1),
                                     torch.nn.Flatten(),
                                     torch.nn.Linear(32,10))
        return self.neural_network

    def display_neural_network(self):
        print ("Neural Network\n", self.neural_network)

    """
        @brief This functions loads the MNIST data from OpenML to be used for training the Neueral Network
    """
    def load_dataset(self):
        self.X, self.y = fetch_openml( "mnist_784", version = 1, return_X_y = True, as_frame = False)
        return (self.X, self.y)

    """
        This function takes in arguments : 
            - a neural network parameter
            - loss function for evaluation
            - epochs for the number of iterations
            - dataloaders for iterating through them and computing the loss values
    
    """
    
    def fit(self, model: torch.nn.Sequential, loss_funct, epochs: int, train_dl: DataLoader, val_dl: DataLoader ):
        if(isinstance(model, torch.nn.Sequential)):
            if(isinstance(train_dl, DataLoader) and isinstance(val_dl, DataLoader)):
                for epoch in range(epochs):
                    model.train()
                    for l_1, (v, w) in enumerate(train_dl):
                        print("type of v is", v.dtype)
                        self.loss_val = loss_funct(model(v.to(torch.float32)), w.to(torch.float32))
                        self.loss_val.backward() 
                        for param in model.parameters():
                            param -= 0.03 * param.grad
                        model.zero_grad()
                        model.eval()

                        with torch.no_grad():
                            self.acc_loss, self.acc_val = 0.0, 0.0
                            for i , (x_1, y_1) in enumerate(val_dl):
                                self.acc_loss +=  loss_funct(model(x_1), y_1)
                                self.acc_val += (torch.argmax(model(x_1),dim =1) == y_1).float().mean()
                return (epoch, (self.acc_loss/(len(val_dl))), (self.acc_val/(len(val_dl))) )

"""
    @brief Class DS inherits the Dataset class from torch.utils.data and stores the training and test data based on the inputs 
"""

class DS(Dataset):

    def __init__(self, x, y):
        self.x = x 
        self.y = y 

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])
    
    def __len__(self):
        return (len(self.x))


"""
    @brief This is the driver code. 
    Creates an object of the model class
    Calls methods to generate a neural network, load dataset and fit function

"""
if(__name__ == "__main__"):

    model = create_model(3,2)

    model.show_model()

    print("Tensor after SGD:\n",model.sgd())

    
    X, y = model.load_dataset()

    sample_size = 5000
    sample_test_size = sample_size * 2 

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, train_size=sample_size, test_size=sample_test_size)

    train_dataset = DS(X_train, Y_train)

    value_dataset = DS(X_test, Y_test)

    train_data_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True, num_workers=10)

    valid_data_loader = DataLoader(value_dataset, batch_size=128, shuffle=False, num_workers=10)

    md = model.generate_neural_network()
    epoch = 10
    model.fit(model=md.parameters(), loss_funct=F.cross_entropy, epochs=epoch, train_dl=train_data_loader, val_dl=valid_data_loader)
    
    print("Model fit as per the parameters. Execution Completed")