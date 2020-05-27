# malaria_detection

## introduction
Diagnose malaria from thin red blood cell using randomly wired Neural network

## Installation 
``` pip install -r requirements.txt ```

## Datasets
you can get dataset from kaggle or from this link 
https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria

to Get Indicies Split data into Train, Validation and Test Data Run This :

``` python splitdata.py [Some Argument]```

| Argument | default | help |
|:---------|:--------|:-----|
|'--img_path'| default='../cell-images-for-detecting-malaria/Fill') | Path of folder that contain dataset |
|'--test_size'| default=0.1) | Size of data Test |
|'--valid_size'| default=0.2) | Size of data Validation |

## Training 

``` python train.py [Some Arguments] ```

| Argument | default | help |
|:---------|:--------|:-----|
|'--channels'| default=3) | Num of Channels |
|'--net_type'| default='4graph') | Num of Graph Use in network |
|'--nodes'| default=10) | Num of Node for every Graph |
|'--graph_model'| default='WS'| Type of Random Graph Use in Network |
|'--K'| default=3| Each node is connected to k nearest neighbors in ring topology|
|'--P'| default=0.5| The probability of rewiring each edge|
|'--img_shape'| default=(64,64)| Size of Resize Image |
|'--Mean_image'| default=[0.485, 0.456, 0.406]| List of Mean Image per Channel |
|'--Var_image'| default=[0.229, 0.224, 0.225]|  List of Variance Image per Channel |
|'--n_epoch'| default=100| Number of Epoch |
|'--batch_size'| default=8 | Size of Batch |
|'--Num_workers'| default=0 | Number of Worker |
|'--img_path'| default='../cell-images-for-detecting-malaria/Fill'| Directory of Dataset|
|'--model_dir'| default='Model/RW4/'| Directory of Graph|
|'--idx_dir'| default='split_random.npy'| Directory of indicies of train, valid and test|
|'--resume'| default=False| Resume |

## Testing
``` python test [Some Argument]```
| Argument | default | help |
|:---------|:--------|:-----|
|'--channels'| default=3) | Num of Channels |
|'--net_type'| default='4graph') | Num of Graph Use in network |
|'--img_shape'| default=(64,64)| Size of Resize Image |
|'--Mean_image'| default=[0.485, 0.456, 0.406]| List of Mean Image per Channel |
|'--Var_image'| default=[0.229, 0.224, 0.225]|  List of Variance Image per Channel |
|'--batch_size'| default=8 | Size of Batch |
|'--Num_workers'| default=0 | Number of Worker |
|'--img_path'| default='../cell-images-for-detecting-malaria/Fill'| Directory of Dataset|
|'--model_dir'| default='Model/RW4/'| Directory of Graph|
|'--best_model'| default='Model/RW4/best_model.pt' | Directory of Best Model Trainer can Get |
|'--idx_dir'| default='split_random.npy'| Directory of indicies of train, valid and test|
|'--resume'| default=False| Resume |
