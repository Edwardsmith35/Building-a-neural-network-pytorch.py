import torch
import torch.nn as nn

X = torch.tensor([[1,2],[3,4],[5,6],[7,8]]).float()
Y = torch.tensor([[3],[7],[11],[15]]).float()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
x, y = X.to(device), Y.to(device)

class MyNeuralNet(nn.Module):
    def __init__(self):
        super().__init__() # take advantage of all the pre-built functionalities that have been written for nn.Module.
        self.input_to_hidden_layer   = nn.Linear(2,8)
        self.hidden_layer_activation = nn.ReLU()
        self.hidden_to_output_layer  = nn.Linear(8,1)
    
    def forward(self, x): # It is mandatory to use forward as the function name, since PyTorch has reserved this function as the method for performing forward-propagation. Using any other name in its place will raise an error.
        x = self.input_to_hidden_layer(x)
        x = self.hidden_layer_activation(x)
        x = self.hidden_to_output_layer(x)
        return x
  
print(nn.Linear(2,7))
# the neural network is initialized with random values every time. If you wanted them to remain the same when executing the code over multiple iterations, you would need to specify the seed using the manual_seed method in Torch as torch.manual_seed(0) just before creating the instance of the class object.
# torch.manual_seed(0)
mynet = MyNeuralNet().to(device)
# NOTE - This line of code is not a part of model building, 
# this is used only for illustration of 
# how to obtain parameters of a given layer
mynet.input_to_hidden_layer.weight
# how to obtain parameters of all layers in a model
mynet.parameters()
# Finally, the parameters are obtained by looping through the generator returned from mynet.parameters(), as follows:
for par in mynet.parameters():
    print(par)

loss_func = nn.MSELoss()
# The loss value of a neural network can be calculated by passing the input values through the neuralnet object and then calculating MSELoss for the given inputs:
_y = mynet(x)
loss_value = loss_func(_y, y) # Also, note that when computing the loss, we always send the prediction first and then the ground truth. This is a PyTorch convention.
print(loss_value)

from torch.optim import SGD
opt = SGD(mynet.parameters(), lr = 0.001)

opt.zero_grad() # flush the previous epoch's gradients
loss_value = loss_func(mynet(X),Y) # compute loss
loss_value.backward() # perform backpropagation
opt.step() # update the weights according to the #gradients computed

loss_history = []
for _ in range(50):
    opt.zero_grad()
    loss_value = loss_func(mynet(X),Y)
    loss_value.backward()
    opt.step()
    loss_history.append(loss_value.item())

# Let’s plot the variation in loss over increasing epochs
import matplotlib.pyplot as plt
x = range(0,50)
#plt.plot(x, loss_history, label='Loss', color='blue')
plt.plot(loss_history)
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.show()