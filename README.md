## Learning Pytorch 

Pipeline: 
- Define the network 
- Define the forward function 
- The backward function is automatically defined for us using `autograd` 

Defining a neural network
Processing inputs and calling backward
Computing the loss
Updating the weights of the network
Updating the weights of the network


Note
- nn.Conv2d will take in a 4D Tensor of nSamples x nChannels x Height x Width.
- If you have a single sample, just use input.unsqueeze(0) to add a fake batch dimension.

