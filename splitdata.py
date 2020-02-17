import numpy as np 

from torchvision import datasets, transforms

import argparse


parser = argparse.ArgumentParser(
    description='Pytorch Split Data Randomly')
parser.add_argument('--img_path', default='../cell-images-for-detecting-malaria/Fill', type=str)
parser.add_argument('--test_size', default=0.1)
parser.add_argument('--valid_size', default=0.2)

args = parser.parse_args()

##Load Data
print('Loading Dataset...')
trans = transforms.Compose([transforms.ToTensor()])
data = datasets.ImageFolder(args.img_path, transform = trans)

##Split Data
print('Split Dataset...')
def random_split(test_size, valid_size, data):
    num_data = len(data)
    indices = list(range(num_data))
    np.random.shuffle(indices)
    valid_split = int(np.floor((valid_size) * num_data))
    test_split = int(np.floor((valid_size+test_size) * num_data))
    valid_idx, test_idx, train_idx = indices[:valid_split], indices[valid_split:test_split], indices[test_split:]
    return valid_idx, test_idx, train_idx

# def kfold_split(num_fold, data):
#     num_data = len(data)
#     indices = list(range(num_data))
#     # np.random.shuffle(indices)
#     num_dat = num_data//2
#     devide = num_dat//num_fold
#     # print(num_dat)
#     fold = []
#     awal = 0
#     for i in range(num_fold):
#         akhir = devide*(i+1)
#         fld = indices[:akhir]
#         fld.extend(indices[num_dat+awal:num_dat:akhir])
#         fold.append(indices[awal:akhir])
#         fold[i].extend(indices[num_dat+awal:num_dat+akhir])
#         awal = akhir
#         # print(fld[i])
#     if num_dat+akhir != num_data:
#         fold[num_fold-1].extend(indices[num_dat+akhir:num_data])
#     return fold

valid_idx, test_idx, train_idx = random_split(args.test_size, args.valid_size, data)

split = [valid_idx, test_idx, train_idx]
# split = kfold_split(5, data)

print('Save Index Dataset...')
np.save('split_random.npy',split, allow_pickle=True)

fold = np.load('split_random.npy', allow_pickle=True)

print(len(fold))
print(len(fold[0]))
print(len(fold[1]))
print(len(fold[2]))
# print(len(fold[3]))
# print(len(fold[4]))
