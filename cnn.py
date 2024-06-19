import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm.notebook import tqdm, trange

# Load the data
mnist_train = datasets.MNIST(root="./datasets", train=True, transform=transforms.ToTensor(), download=True)
mnist_test = datasets.MNIST(root="./datasets", train=False, transform=transforms.ToTensor(), download=True)
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=100, shuffle=False)

## Training
# Instantiate model  
#model = MNIST_CNN()  # <---- change here
### YOUR CODE HERE ###
#imports
import numpy as np
import torch
import torch.nn.functional as F


#step1: take a random image
# Create a random image input tensor
x_cnn = torch.randn(100, 3, 32, 32)
print("Shape of the image tensor: {}".format(x_cnn.shape))

#step2:First Convolution Layer
W1 = torch.randn(32, 3, 3, 3)/np.sqrt(3*3*3)
W1.requires_grad_()
b1 = torch.zeros(32, requires_grad=True)

# Apply convolutional and ReLU
conv1_preact = F.conv2d(x_cnn, W1, bias=b1, stride=1, padding=1)
conv1 = F.relu(conv1_preact)

# Print input/output shape
print("First Convolution output shape: {}".format(conv1.shape))

#step3:First Convolution Layer
W2 = torch.randn(32, 32, 3, 3)/np.sqrt(32*3*3)
W2.requires_grad_()
b2 = torch.zeros(32, requires_grad=True)

# Apply convolutional and ReLU
conv2_preact = F.conv2d(conv1, W2, bias=b2, stride=1, padding=1)
conv2 = F.relu(conv2_preact)

# Print input/output shape
print("Second Convolution output shape: {}".format(conv2.shape))


#step4: Pooling
max_pool2 = F.max_pool2d(conv2, kernel_size=2)
print("Shape of conv2 feature maps after max pooling: {0}".format(max_pool2.shape))


#step5: 3rd convolutional layer
# 3rd layer variables
W3 = torch.randn(64, 32, 3, 3)/np.sqrt(32*3*3)
W3.requires_grad_()
b3 = torch.zeros(64, requires_grad=True)

# Apply convolutional and ReLU
conv3 = F.relu(F.conv2d(max_pool2, W3, bias=b3, stride=1, padding=1))
print("Third Convolution output shape: {}".format(conv3.shape))


#step6: 4th convolutional layer
W4 = torch.randn(64, 64, 3, 3)/np.sqrt(64*3*3)
W4.requires_grad_()
b4 = torch.zeros(64, requires_grad=True)

# Apply convolutional and ReLU
conv4 = F.relu(F.conv2d(conv3, W4, bias=b4, stride=1, padding=1))
print("Fourth Convolution output shape: {}".format(conv4.shape))

#step7: Pooling
max_pool4 = F.max_pool2d(conv4, kernel_size=2)
print("Shape of conv2 feature maps after max pooling: {0}".format(max_pool4.shape))


#step8: flatten 2D into 1D
# Flatten convolutional feature maps into a vector
h_flat = torch.flatten(max_pool4, 1)

# Print output shape
print("Flatten shape: ", h_flat.shape)

#step9: Fully connected 256
fc1 = nn.Linear(8*8*64, 256)
h_fc1 = F.relu(fc1(h_flat))
print("FC1 output shape: ", h_fc1.shape)


#step10
fc2 = nn.Linear(256, 10)
output = F.softmax(fc2(h_fc1), dim=1)
print("Output shape (after softmax): ", output.shape)


# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # <---- change here

# Iterate through train set minibatchs 
for epoch in trange(3):  # <---- change here
    for images, labels in tqdm(train_loader):
        # Zero out the gradients
        optimizer.zero_grad()

        # Forward pass
        x = images  # <---- change here 
        y = model(x)
        loss = criterion(y, labels)
        # Backward pass
        loss.backward()
        optimizer.step()

## Testing
correct = 0
total = len(mnist_test)

with torch.no_grad():
    # Iterate through test set minibatchs 
    for images, labels in tqdm(test_loader):
        # Forward pass
        x = images  # <---- change here 
        y = model(x)

        predictions = torch.argmax(y, dim=1)
        correct += torch.sum((predictions == labels).float())

print('Test accuracy: {}'.format(correct/total))