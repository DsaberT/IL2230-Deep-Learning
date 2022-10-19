import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from torch.profiler import profile, record_function, ProfilerActivity

from torchvision import datasets
import os

"""1. LOAD AND NORMALIZE CIFAR10"""

#transform = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])

data_transforms = {
    'train': transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    'val': transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
}


data_dir = 'hymenoptera_data'

trainset = datasets.ImageFolder(os.path.join(data_dir, 'train'),
                                          data_transforms['train'])
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                             shuffle=True, num_workers=0) #0 or 4?

testset = datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                          data_transforms['val'])
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=True, num_workers=0) #0 or 4?

#trainset_sizes = len(trainset)
#testset_sizes = len(testset)



classes = ('ants','bees')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""1.1 SHOW SOME TRAINING IMAGES JUST FOR FUN"""
# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

"""2. DEFINE A CONVOLUTIONAL NEURAL NETWORK"""
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 12, 5)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(12, 30, 5)
        #self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(30, 120, 5)
        self.conv4 = nn.Conv2d(120, 300, 5,1,2)
        self.fc1 = nn.Linear(300, 84)
        self.fc2 = nn.Linear(84, 2)


    def forward(self, x):


        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))

        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x

net = Net()

"""3. DEFINE A LOSS FUNCTION AND OPTIMIZER"""
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

"""4. TRAIN THE NETWORK"""
for epoch in range(20):  # loop over the dataset multiple times


    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics

        running_loss += loss.item()
        if i % 50 == 49:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
              (epoch + 1, i + 1, running_loss / 50))
            running_loss = 0.0



print('Finished Training')
"""
with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("model_interference"):
        net(inputs)
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
with profile(activities=[ProfilerActivity.CPU],profile_memory=True, record_shapes=True) as prof:
    net(inputs)
print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))
"""

"""On the Whole Data set"""
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %.2f %%' % (
    100 * correct / total))

"""Accuracy per Class"""

# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print("Accuracy for class {:5s} is: {:.1f} %".format(classname, accuracy))
