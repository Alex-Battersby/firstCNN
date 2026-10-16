import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim

from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import time
import os
import copy
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


num_epochs = 50
learning_rate = 0.01
batch_size = 4

compose = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5,0.5,0.5))])
cifarTrain = torchvision.datasets.CIFAR10('./cifar', train=True, transform=compose, download=False)
cifarTest = torchvision.datasets.CIFAR10('./cifar', train=False, transform=compose, download=False)

trainSet = DataLoader(cifarTrain, batch_size=batch_size, shuffle=True)
testSet = DataLoader(cifarTest, batch_size=batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 9)
        #3x32x32 to 8x24x24
        self.pool = nn.MaxPool2d(2, 2)
        #24x24 to 12x12
        self.conv2 = nn.Conv2d(8, 25, 5)
        #12x12 to 8x8
        self.fc1 = nn.Linear(25*4*4, 150)
        self.fc2 = nn.Linear(150, 75)
        self.fc3 = nn.Linear(75, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 25*4*4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = Model()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(trainSet):
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1)%2000 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, step {i+1}/{len(trainSet)}, loss: {loss.item():.4f}')

print('Finished training')

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_correct_class = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in testSet:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        n_samples += images.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if label == pred:
                n_correct_class[label] += 1
            n_class_samples[label] +=1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of network: {acc}%')
    for i in range(10):
        acc = 100.0 * n_correct_class[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc}%')