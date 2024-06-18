import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import datasets, transforms
from tqdm.notebook import tqdm

# Load the data
mnist_train = datasets.MNIST(root="./datasets", train=True, transform=transforms.ToTensor(), download=True)
mnist_test = datasets.MNIST(root="./datasets", train=False, transform=transforms.ToTensor(), download=True)

train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=64, shuffle=False)


#Model Parameters
input_size = 784
hidden_size = 500
output_size = 10

#Xavier Initialization of weights and bias 
W1 = torch.randn(input_size, hidden_size) / np.sqrt(input_size)
W1.requires_grad_()
b1 = torch.zeros(hidden_size, requires_grad=True)
W2 = torch.randn(hidden_size, output_size) / np.sqrt(hidden_size)
W2.requires_grad_()
b2 = torch.zeros(output_size, requires_grad=True)


epochNum = 5
#optimizer
optimizer = torch.optim.SGD([W1,b1, W2, b2], lr=0.01)


##Training 
for epoch in range(epochNum):
    for images, labels in tqdm(train_loader):
        #Flatten the image
        images = images.view(-1, input_size)
        
        #forward pass
        y = torch.matmul(images, W1) + b1
        hidden_layer = F.relu(y)
        output_layer = torch.matmul(hidden_layer, W2) + b2
        
        #loss function calc
        loss = F.cross_entropy(output_layer, labels)

        #backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f'Epoch [{epoch+1}/{epochNum}], Loss: {loss.item():.4f}')


## Testing
correct = 0
total = len(mnist_test)

with torch.no_grad():
    # Iterate through test set minibatchs 
    for images, labels in tqdm(test_loader):
        #Flatten the image
        images = images.view(-1, input_size)
        
        #forward pass
        y = torch.matmul(images, W1) + b1
        hidden_layer = F.relu(y)
        output_layer = torch.matmul(hidden_layer, W2) + b2
        
    
        # Use the output of the second layer for predictions
        predictions = torch.argmax(output_layer, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    
print('Test accuracy: {}'.format(correct/total))