import os
import time
import math
import random
import numpy as np
import pandas as pd
from pathlib import Path
import glob
from PIL import Image, ImageEnhance, ImageOps
import natsort
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
#import matplotlib.pyplot as plt
import time
import copy


path_dataset = os.path.join(os.getcwd(), 'dataset')
path_train = os.path.join(path_dataset, 'train')
path_train = os.path.join(path_dataset, 'val')
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join(path_dataset, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=8,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time() # Set current(start) time

    best_model_wts = copy.deepcopy(model.state_dict()) # model.state_dict() 전체 객체 복사 best_model_wts와 model.state_dict는 서로 영향
                                                    # 받지 않음.
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients, 역전파 수행전 기울기(변화도 0으로)
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'): # grad 변화 가능
                    outputs = model(inputs) # model에 입력 텐서 넣음.
                    _, preds = torch.max(outputs, 1) # preds, 예측값,
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward() # 역전파
                        optimizer.step() # 파라미터 업데이트

                # statistics
                running_loss += loss.item() * inputs.size(0) # loss.item() : gets the a scalar value held in the loss.
                                                             # inouts.size(0) : batch size
                # running_loss = running_loss + loss.item() * input.size(0)
                
                running_corrects += torch.sum(preds == labels.data) # 맞은 개수

            epoch_loss = running_loss / dataset_sizes[phase] #지정된 배치 크기만큼해서 전체 데이터셋 돌고 난 뒤 얻은 loss
            epoch_acc = running_corrects.double() / dataset_sizes[phase] # 전체 acc

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            # val을 할 차례
            if phase == 'val' and epoch_acc > best_acc: 
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())



    time_elapsed = time.time() - since # 시간 얼마나 지났니
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model #가장 좋은 모델의 파라미터를 가지고 리턴해줌.

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
		
model_resnet = models.resnet18(pretrained=True) # resnet18 사용
model_next = models.resnext50_32x4d(pretrained=True)

# feature of fc layer each model
num_ftrs = model_next.fc.in_features
print(model_next.fc)

model_next.fc = nn.Linear(num_ftrs, 196) # 196개 클래스 이므로

model_next = model_next.to(device) # tensor 변환

criterion = nn.CrossEntropyLoss() # 이 로스 함수 사용

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_next.parameters(), lr=0.001, momentum=0.9)  #optimizer 설정

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1) # ㅇㅇ
model_next = train_model(model_next, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=100)


torch.save(model_ft, './test/save_next.pth')
                       
