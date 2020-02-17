## Inisiasi
import numpy as np
#import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torchvision import transforms, datasets, models
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from model import CNN
import argparse

import random

parser = argparse.ArgumentParser(
    description='Pytorch Imagenet Training')
parser.add_argument('--channels', default=3)
parser.add_argument('--net_type', default='4graph')
parser.add_argument('--nodes', default=10)
parser.add_argument('--graph_model', default='WS')
parser.add_argument('--K', default=3, help='Each node is connected to k nearest neighbors in ring topology')
parser.add_argument('--P', default=0.5, help='The probability of rewiring each edge')
parser.add_argument('--img_shape', default=(64,64))
parser.add_argument('--Mean_image', default=[0.485, 0.456, 0.406])
parser.add_argument('--Var_image', default=[0.229, 0.224, 0.225])
parser.add_argument('--n_epoch', default=100)
parser.add_argument('--batch_size', default=8)
parser.add_argument('--Num_workers', default=0)
parser.add_argument('--seed', default=None)
parser.add_argument('--world-size', default=1, type=int)
parser.add_argument('--rank', default=0, type=int)
parser.add_argument('--img_path', default='../cell-images-for-detecting-malaria/Fill', type=str)
parser.add_argument('--model_dir', default='Model/RW4/', type=str)
parser.add_argument('--idx_dir', default='split_random.npy', type=str)
parser.add_argument('--resume', default=False, help='resume')

args = parser.parse_args()

trans = transforms.Compose([transforms.Resize(args.img_shape),
                            transforms.ToTensor(),
                           transforms.Normalize(args.Mean_image,args.Var_image)])

# define samplers for obtaining training and validation batches
split = np.load(args.idx_dir, allow_pickle=True)

train_sampler = SubsetRandomSampler(split[2])
valid_sampler = SubsetRandomSampler(split[0])
test_sampler = SubsetRandomSampler(split[1])

# prepare data loaders (combine dataset and sampler)
print('Loading Data...')
train_data = datasets.ImageFolder(args.img_path,transform=trans)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
    sampler=train_sampler, num_workers=args.Num_workers)

validation_data = datasets.ImageFolder(args.img_path,transform=trans)
valid_loader = torch.utils.data.DataLoader(validation_data, batch_size=args.batch_size, 
    sampler=valid_sampler, num_workers=args.Num_workers)
datasets
test_data = datasets.ImageFolder(args.img_path,transform=trans)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, 
    sampler=test_sampler, num_workers=args.Num_workers)


# Model
print('Creating Model...')
model = CNN(args)

## For ResNet50
# model = models.resnet50(pretrained=True)

# for param in model.parameters():
#     param.requires_grad = False

# model.fc = nn.Linear(2048, 2, bias=True)

# fc_parameters = model.fc.parameters()

# for param in fc_parameters:
#     param.requires_grad = True
    
print(model) # net architecture

#Training Model
print('Training...')
use_cuda = torch.cuda.is_available()
if use_cuda:
    model = model.cuda() 
    print('Use GPU for Running Pogram')
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

def train(n_epochs, model, optimizer, criterion, use_cuda, save_path, lanjutan= None, Log=50):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf
    all_loss_train = []
    all_loss_valid = []
    all_acc_train = []
    all_acc_valid = []
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        correct_train = 0.
        total_train = 0.    
        correct_valid = 0.
        total_valid = 0.
        
        try :
            model.load_state_dict(torch.load(lanjutan))
            print("Continuing....")
        except :
            pass

        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()

            # initialize weights to zero
            optimizer.zero_grad()
            
            output = model(data)
            
            # calculate loss
            loss = criterion(output, target)
            
            # back prop
            loss.backward()
            
            # grad
            optimizer.step()
            
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            
            pred = output.data.max(1, keepdim=True)[1]
            
            # compare predictions to true label
            correct_train += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
            
            total_train += data.size(0)
            
        #save loss to analys
        all_loss_train.append(train_loss)
        all_acc_train.append(correct_train/total_train)
        
        ######################    
        # validate the model #
        ######################model_dir
        model.eval()
        for batch_idx, (data, target) in enumerate(valid_loader):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## update the average validation loss
            output = model(data)
            loss = criterion(output, target)
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
            
            pred = output.data.max(1, keepdim=True)[1]
            
            # compare predictions to true label
            correct_valid += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
            
            total_valid += data.size(0)
        
        #save loss to analys
        all_loss_valid.append(valid_loss)
        all_acc_valid.append(correct_valid / total_valid)
    
        ## TODO: save the model if validation loss has decreased
        if valid_loss < valid_loss_min:
            torch.save(model.state_dict(), '{}best_model.pt'.format(save_path))
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,
                                                                                            valid_loss))
            valid_loss_min = valid_loss

        print('Epoch: {}'.format(epoch+1))
        print('\tTrain Loss: {0:.2f} | Train Acc: {1:.2f}%'.format(train_loss, (correct_train/total_train)*100))
        print('\t Val. Loss: {0:.2f} |  Val. Acc: {1:.2f}%'.format(valid_loss, (correct_valid / total_valid)*100))
    
        if (epoch+1) % Log == 0:
            torch.save(model.state_dict(), '{}Epoch:_{}-Acc:_{}.pt'.format(save_path, epoch+1, 100. * correct_train / total_train))
    
    np.save('{}Log/loss_train.npy'.format(save_path), all_loss_train)
    np.save('{}Log/loss_valid.npy'.format(save_path), all_loss_valid)
    np.save('{}Log/acc_valid.npy'.format(save_path), all_acc_valid)
    np.save('{}Log/acc_train.npy'.format(save_path), all_acc_train)
    print("Saved Log")
    # return trained model
    return model
        
train(args.n_epoch, model, optimizer, criterion, use_cuda, args.model_dir)