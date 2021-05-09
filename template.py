from __future__ import print_function, division
import datetime
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms,utils
import time
import os
import copy
import torch.nn as nn
import torch.nn.functional as F


# TODO: Implement a convolutional neural network (https://pytorch.org/tutorials/recipes/recipes/defining_a_neural_network.html)
class Net(nn.Module):
    """
    Input - 1x32x32
    Output - 10
    """
    def __init__(self):
        super(Net, self).__init__()
        # TODO: Initialize layers
        # CONV1-CONV2-POOL1-CONV3-CONV4-POOL2-FC1-FC2-FC3

        self.conv1 = nn.Conv2d(3,32,8,1,0)
        self.conv2 = nn.Conv2d(32,64,4,1,1)
        self.conv3 = nn.Conv2d(64,64,3,1,0)
        self.conv4 = nn.Conv2d(64,16,3,1,1)


        self.pool1 = nn.MaxPool2d(2,2,0)
        self.pool2 = nn.MaxPool2d(2,2,0)

        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,90)
        self.fc3 = nn.Linear(90,10)

    def forward(self, img):

        # TODO: Implement forward pass
        x = img
        x = F.relu(self.conv1(x))
        #print("CONV1",x.size())
        #x = self.pool1(x)
        #print("POOL1",x.size())
        x = F.relu(self.conv2(x))
        #print("CONV2",x.size())
        x = self.pool1(x)
        #print("POOL1",x.size())
        x = F.relu(self.conv3(x))
        #print("CONV3",x.size())
        #x = self.pool3(x)
        #print("POOL3",x.size())
        x = F.relu(self.conv4(x))
        #print("CONV4",x.size())
        #x = F.relu(self.conv5(x))
        #print("CONV5",x.size())
        x = self.pool2(x)
        #print("POOL2",x.size())
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        #print("FC1",x.size())
        x = F.relu(self.fc2(x))
        #print("FC2",x.size())
        x = F.relu(self.fc3(x))
        #print("FC3",x.size())


        return x

# TODO: You can change these data augmentation and normalization strategies for
#  better training and testing (https://pytorch.org/vision/stable/transforms.html)
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((32,32)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Dataset initialization
data_dir = 'data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'test']} # Read train and test sets, respectively.

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=0) for x in ['train', 'test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}

class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Set device to "cpu" if you have no gpu

# TODO: Implement training and testing procedures (https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
def train_test(model, criterion, optimizer, scheduler, num_epochs=25):
    print("begin training",datetime.datetime.now())
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        print("epoch: ",epoch,datetime.datetime.now())
        running_loss = 0.0
        for i, data in enumerate(dataloaders['train'], 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training',datetime.datetime.now())
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(dataloaders['train'], 0):
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Training accuracy: %d %%' % (100 * correct / total))

    # overall testing correct rate
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(dataloaders['test'], 0):
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Testing accuracy: %d %%' % (100 * correct / total))

    # count testing predictions for each class
    correct_pred = {classname: 0 for classname in class_names}
    total_pred = {classname: 0 for classname in class_names}
    with torch.no_grad():
        for i, data in enumerate(dataloaders['test'], 0):
            images, labels = data
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[class_names[label]] += 1
                total_pred[class_names[label]] += 1

    # print accuracy for each class
    print("Testing accuracy (each class): ")
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print("{:1s}: {:.1f}%;  ".format(classname, accuracy), end=' ')
        if classname == "5":
            print()
    print()

    return None

model_ft = Net() # Model initialization
model_ft = model_ft.to(device) # Move model to cpu
criterion = nn.CrossEntropyLoss() # Loss function initialization
# TODO: Adjust the following hyper-parameters: learning rate, decay strategy, number of training epochs.
optimizer_ft = optim.Adam(model_ft.parameters(), lr=2e-4) # Optimizer initialization

exponen_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1)
epoch = 30

for n in range(30):
    print(datetime.datetime.now())
    train_test(model_ft, criterion, optimizer_ft, exponen_lr_scheduler, num_epochs=1)

PATH = './5.pth'
torch.save({
        'epoch': epoch,
        'model_state_dict': model_ft.state_dict(),
        'optimizer_state_dict': optimizer_ft.state_dict()
        }, PATH)
