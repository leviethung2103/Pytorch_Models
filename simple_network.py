import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10) # 10 classes

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)

# number of learnable parameters
params = list(net.parameters())
print("Number of learnable parameters:",len(params))
print(params[0].size())  # conv1's .weight


# use random input 
# letnet: 32x32

input = torch.randn(1,1,32,32)
out = net(input)
print ("Input shape:",input.shape)
print ("out shape:",out.shape)

# Zero the gradient buffers of all parameters and backprops with random gradients:
net.zero_grad()
# backprops with random gradients
out.backward(torch.randn(1, 10))


# forward pass 
output = net(input)
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)

# backward 
print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

# tensor([-0.0656,  0.0709,  0.0126, -0.0690, -0.0692, -0.0218])

# clear all the gradients 
net.zero_grad()
print('conv1.bias.grad Zero the gradient buffers')
print(net.conv1.bias.grad)

# after backprogration 
loss.backward()
print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

# Update the weight - simple python code
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)


# library
# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)
optimizer.step()    # Does the update

# Observe how gradient buffers had to be manually set to zero using optimizer.zero_grad(). This is because gradients are accumulated as explained in the Backprop section.