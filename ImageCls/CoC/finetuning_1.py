#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import time
import math
import random
import time
import copy
import glob
import natsort

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from PIL import Image, ImageEnhance, ImageOps
from pathlib import Path
import torchvision
from torchvision import datasets, models, transforms
from torchsummary import summary
#import matplotlib.pyplot as plt

from lr_scheduler import WarmUpLR
from criterion import LSR

# evaluation
from sklearn.metrics import accuracy_score, precision_score, precision_recall_curve
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score


# ## Training Model (fine-tuned)

# In[2]:


# def get_auc(labels, preds):
#     labels = labels.numpy()
#     preds = preds.numpy()
#     roc_auc_score(y_true, y_scores)

def split_weights(net):
    """split network weights into to categlories,
    one are weights in conv layer and linear layer,
    others are other learnable paramters(conv bias, 
    bn weights, bn bias, linear bias)
    Args:
        net: network architecture
    
    Returns:
        a dictionary of params splite into to categlories
    """

    decay = []
    no_decay = []

    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            decay.append(m.weight)

            if m.bias is not None:
                no_decay.append(m.bias)
        
        else: 
            if hasattr(m, 'weight'):
                no_decay.append(m.weight)
            if hasattr(m, 'bias'):
                no_decay.append(m.bias)
        
    assert len(list(net.parameters())) == len(decay) + len(no_decay)

    return [dict(params=decay), dict(params=no_decay, weight_decay=0)]

def train_model_trick(model, criterion, optimizer, scheduler,warmup_scheduler, num_epochs=60):
    since = time.time() # Set current(start) time

    best_model_wts = copy.deepcopy(model.state_dict())                              
    best_acc = 0.0


    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                if epoch > 8:
                    scheduler.step(epoch)
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                if epoch <= 8:
                    warmup_scheduler.step()
                   
                inputs = inputs.to(device)
                labels = labels.to(device)

                
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'): 
                    outputs = model(inputs) 
                    _, preds = torch.max(outputs, 1)
                    
                    #precision, recall, _ = precision_recall_curve(labels, preds)
                    #fpr, tpr, _ = roc_curve(labels, preds)
                    #
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward() 
                        optimizer.step() 

                # statistics
                running_loss += loss.item() * inputs.size(0) # loss.item() : gets the a scalar value held in the loss.
                                                             # inouts.size(0) : batch size
                # running_loss = running_loss + loss.item() * input.size(0)
                
                running_corrects += torch.sum(preds == labels.data)
                
            epoch_loss = running_loss / dataset_sizes[phase] 
            epoch_acc = running_corrects.double() / dataset_sizes[phase] 
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
  
            if phase == 'val' and epoch_acc > best_acc: 
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                


    time_elapsed = time.time() - since 
    print('Traiwqning complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model 


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
        
def val_model(path):
    data_transforms = transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229,0.224,0.225])
    ])
    
    # model load
    model_ft = torch.load(path, map_location="cuda:1")
    
    # test images load
     # for val dataset
    path_test = os.path.join(os.getcwd(), 'val')
    image_list = natsort.natsorted(os.listdir(path_test))

    labels = []
    predictions = []
    outputs = []

    # test all images
    for i, imgs in enumerate(image_list):
        labels.append(int(imgs))
        pth_img = os.path.join(path_test, imgs)
        path_img = glob.glob(f'{pth_img}/*.jpg')
        for img in path_img:
            image = Image.open(img)
            imgblob = data_transforms(image)

            imgblob.unsqueeze_(dim=0)
            imgblob = Variable(imgblob)
            imgblob = imgblob.cuda(device)

            torch.no_grad()
            output = model_ft(imgblob)
            final = output

            prediction = int(torch.max(final.data, 1)[1].cpu().numpy())
            prediction = prediction +1

            predictions.append(prediction)
            if i % 50 ==0:
                print(f'======{i} th images predicted....=====>')
                
    return model, predictions, labels
    
def test_model(path):
    data_transforms = transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229,0.224,0.225])
    ])
    
    # model load
    model_ft = torch.load(path, map_location="cuda:1")
    
    # model custimized
    #modules = list(model_ft.children())[:-2]  # remove fc layer
    #model_ft = nn.Sequential(*modules)
    model_ft = nn.Linear(num_ftrs, 9)
    
    path_test = os.path.join(os.getcwd(), 'val') 
    image_list = natsort.natsorted(os.listdir(path_test))

    predictions = []
    outputs = []

    for i, imgs in enumerate(image_list):
        path_img = os.path.join(path_test, imgs)
        image = Image.open(path_img)
        imgblob = data_transforms(image)

        imgblob.unsqueeze_(dim=0)
        imgblob = Variable(imgblob)
        imgblob = imgblob.cuda(device)

        torch.no_grad()
        output = model_ft(imgblob)
        final = output
        prediction = int(torch.max(final.data, 1)[1].cpu().numpy())
        prediction = prediction +1

        #predictions.append(prediction)
        if i % 300 ==0:
            print(f'======{i} th images predicted....=====>')

    return model, predictions

    


# ## Data augmentation

# In[8]:


import matplotlib.pyplot as plt


# In[6]:


path  = './cohack/train'
class_name = os.listdir(path)


# In[7]:


print(class_name)
class_num = []
for idx in class_name:
    path2 = os.path.join(path, idx)
    path_imgs = os.listdir(path2)
    class_num.append(len(path_imgs))
print(class_num)


# In[8]:


def get_bar(y, x_label):
    x = np.arange(len(y))
    plt.title("class numbers")
    plt.bar(x, y)
    plt.xticks(x, x_label)
    plt.yticks(sorted(y))
    plt.xlabel('classes')
    plt.ylabel("counts")
    plt.figure(figsize=(16, 5))
    


# In[ ]:





# ## Model training

# In[3]:



if __name__ == "__main__":
    
    # load dataset
    path_dataset = os.path.join(os.getcwd(), 'cohack')
    path_train = os.path.join(path_dataset, 'train')
    print(path_train)
    
    # model augument
    data_transforms = {
        'train': transforms.Compose([
            #transforms.Random),
            transforms.RandomRotation(15),
            #transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            #transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4,hue=0.1),
            #agu.ImageNetPolicy(),
            transforms.Resize([224,224]),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize([224,224]),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])}
    
    # Data parallalization
    image_datasets = {x: datasets.ImageFolder(os.path.join(path_dataset, x),
                                            data_transforms[x])
                    for x in ['train','val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,
                                                shuffle=True, num_workers=4)
                for x in ['train','val']}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train','val']} #628
    class_names = image_datasets['train'].classes  
    print(class_names)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model_ft = models.resnet152(pretrained=True)
    num_ftrs = model_ft.fc.in_features  
    
    # model custimized
    #modules = list(model_ft.children())[:-1]  # remove fc, (avg)pooling layer
    #model_ft = nn.Sequential(*modules)
    
    # model fine-tuned
# model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, 2048),nn.Linear(2048,9))
    model_ft.fc = nn.Linear(num_ftrs, 9)
    #model_ft.avgpool = nn.AvgPool2d(kernel_size=(1,1))
    model_ft = model_ft.to(device)
    
    # define loss
    criterion = nn.CrossEntropyLoss() 
    #criterion = LSR()
    params = split_weights(model_ft)
    iter_per_epoch = len(dataloaders['train'])

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params, lr=1e-4, momentum=0.9) 
    # add scheduler 
    warmup_scheduler = WarmUpLR(optimizer_ft, iter_per_epoch * 5)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)
    model_ft = train_model_trick(model_ft,criterion, optimizer_ft, exp_lr_scheduler,warmup_scheduler, num_epochs=30)


    # save the fine-tuned model
    torch.save(model_ft, './resnet152_val.pth')


# ## Fine-tuned Model Load

# ## Test Model

# In[4]:


# test_model('./test_resnet152.pth')


# In[5]:


# model = torch.load('./test_resnet152.pth')
# model.eval()


# In[ ]:





# In[ ]:





# In[6]:


preds = torch.tensor([3, 3, 0, 0, 3, 3, 0, 3, 0, 3, 0, 3, 0, 0, 0, 0], device='cuda:1')
labels = torch.tensor([6, 8, 5, 5, 3, 7, 5, 8, 2, 7, 2, 8, 3, 0, 3, 8], device='cuda:1')


# In[8]:


len(torch.eq(preds,labels))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




