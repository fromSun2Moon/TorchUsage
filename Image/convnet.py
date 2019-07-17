import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# run the tensorboard
writer = SummaryWriter()

# Hyper params

num_epochs = 10
num_classes = 10
batch_size = 64
learning_rate = 0.0008

# view


def imgshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, ax(1, 2, 0)))


# CIFAR dataset
 # pytorch의 경우 torchvision에서 데이터셋을 제공
 # original batch size 10000 x 3072 (3*32*32) matrix
""" class label explain
airplane : 0
automobile : 1
bird : 2
cat : 3
deer : 4
dog : 5
frog : 6
horse : 7
ship : 8
truck : 9
"""
train_dataset = torchvision.datasets.CIFAR10(
    root='././data/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.CIFAR10(
    root='././data/', train=False, transform=transforms.ToTensor())

# Data


train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.layer1 = nn.Sequential(
            # conv 2d input , output , kernel size , stride, padding
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64,128, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.fc = nn.Linear(11*11*128, num_classes) # kernel size => -2 / final +1

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)  # 7*7*32
        out = self.fc(out)
        return out


model = ConvNet(num_classes).to(device)

# loss and optimizer

criterion = nn.CrossEntropyLoss()  # model 호출
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# Train the model

total_step = len(train_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], step[{}/{}], Loss: {:.4f}'
                  .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

            writer.add_scalar('Loss', loss.item(), i)

# test the model

model.eval()  # dropout , batch normalization out
with torch.no_grad():
    correct = 0
    total = 0

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)

        _, preds = torch.max(output.data, 1)  # Tensor, dim (32개의 batch의 경우
                                                         # 32개마다의 max value와 인덱스가 반환)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    print('Test accuracy of the model on the 10000 test images {} %'.format(100 * (correct / total)))
