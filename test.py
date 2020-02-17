#Test Model
import numpy as np
import torch
from torch import nn, optim
from torchvision import transforms, datasets, models
from torch.utils.data.sampler import SubsetRandomSampler

from model import CNN

import torch.nn.functional as F

import argparse

from sklearn.metrics import confusion_matrix, classification_report

parser = argparse.ArgumentParser(
    description='Pytorch Imagenet Training')
parser.add_argument('--channels', default=3)
parser.add_argument('--net_type', default='4graph')
parser.add_argument('--batch_size', default=1)
parser.add_argument('--img_shape', default=(64,64))
parser.add_argument('--Mean_image', default=[0.485, 0.456, 0.406])
parser.add_argument('--Var_image', default=[0.229, 0.224, 0.225])
parser.add_argument('--Num_workers', default=0)
parser.add_argument('--img_path', default='../cell-images-for-detecting-malaria/Fill', type=str)
parser.add_argument('--model_dir', default='Model/RW4/', type=str)
parser.add_argument('--best_model', default='Model/RW4/best_model.pt', type=str)
parser.add_argument('--idx_dir', default='split_random.npy', type=str)
parser.add_argument('--resume', default=True, help='resume')

args = parser.parse_args()

print('Loading Data...')
test_transforms = transforms.Compose([transforms.Resize(args.img_shape),
                                      transforms.ToTensor(),
                                      transforms.Normalize(args.Mean_image,args.Var_image)])

split = np.load(args.idx_dir, allow_pickle=True)
test_sampler = SubsetRandomSampler(split[1])

test_data = datasets.ImageFolder(args.img_path, transform=test_transforms)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, 
    sampler=test_sampler, num_workers=args.Num_workers)


print('Loading Model')
model = CNN(args)

## For ResNet50
# model = models.resnet50(pretrained=True)

# for param in model.parameters():
#     param.requires_grad = False

# model.fc = nn.Linear(2048, 2, bias=True)

# fc_parameters = model.fc.parameters()

# for param in fc_parameters:
#     param.requires_grad = True

model.load_state_dict(torch.load(args.best_model, map_location=torch.device('cpu')))

model.eval()

use_cuda = torch.cuda.is_available()
if use_cuda:
    model = model.cuda() 
    print('Use GPU for Running Pogram')
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

print('Testing...')
def test(model, criterion, use_cuda):

    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    for batch_idx, (data, target) in enumerate(test_loader):
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss 
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)
            
    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (100. * correct / total, correct, total))

    #np.save('Model_CNN_1/Log/Test_acc.npy' (correct / total))


# test(model, criterion, use_cuda)

def predict(model, use_cuda, class_names=['Parasitized','Uninfected']):
    predicted = []
    true = []
    probabiliti = []
    for batch_idx, (data, target) in enumerate(test_loader):
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            hasil = F.softmax(model(data)).numpy()
        # print(hasil)
        probabiliti.append(hasil[0])
        idx = np.argmax(hasil[0])
        predicted.append(class_names[idx])
        true.append(class_names[np.asscalar(target)])
    return predicted, true, probabiliti

y_pred, y_true, y_prob = predict(model, use_cuda)

print(y_prob)
y = [y_pred, y_true]
np.save('y.npy', y)
np.save('probabilities.npy',y_prob)

y_new = np.load('y.npy', allow_pickle=True)


# Menghitung Jumlah Parameter
# tensor_dict = torch.load('model.dat', map_location='cpu') # OrderedDict

print(confusion_matrix(y_new[1], y_new[0], labels=['Parasitized','Uninfected']),'\n')
print(classification_report(y_new[1], y_new[0], labels=['Parasitized','Uninfected']))


