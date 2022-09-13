import os

import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from SIAM.datasets import SUNAttributesDataset, SoNDataset, my_collate
from SIAM.custom_transforms import ToTensor, Rescale, RandomCrop, VerticalFlip, Rotate
from SIAM.models import NetSUNTop, NetSoNTop
from SIAM.utils import load_SoN_images

gpu_no = 0  # Set to False for cpu-version
freeze_basenet = True
freeze_sun_net = True
# Directory to save the model in
net_folder = 'sun_son'
out_net_name = 'last_model_finetuned.pt'

# Directory of the init model. init_net_name = None for starting from scratch
init_net_folder = 'models'
init_net_name = None

# Data paths
SoN_dir = "../../../data/datasets/SoN/"
SUN_dir = "../../../data/datasets/SUNAttributes/"

# Images are downloaded and stored in a folder numbered 1-10
# The votes.tsv file contains URIs which can be used to download the images
# Because of license constraints we cannot re-upload these images, so the user has to dowload them separately
image_folders = [str(num) for num in range(1,11)]

#warm_start = False
epochs_only_sun = 0
epochs_only_son = 0
init_lr = 1e-3
reduce_lr = [10,20]
n_epochs = 200

if type(gpu_no) == int:
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


if not os.path.exists(net_folder):
    os.mkdir(net_folder)

# Read images and labels of ScenicOrNot (SoN) dataset
im_paths, son_avg, son_var = load_SoN_images(SoN_dir, image_folders)
labels_son = np.array([son_avg,son_var]).transpose()

# Build train-val-test sets
im_paths_son_train = im_paths[0:150000]
labels_son_train = labels_son[0:150000,:]
im_paths_son_val = im_paths[150000:155000]
labels_son_val = labels_son[150000:155000,:]
im_paths_son_test = im_paths[155000:160000]
labels_son_test = labels_son[155000:160000,:]

composed = transforms.Compose([Rescale((500,500)),Rotate(5),RandomCrop(450),VerticalFlip(),ToTensor()])
dataset_son_train = SoNDataset(im_paths_son_train,labels_son_train,['Average','Variance'],tr=composed)
dataset_son_val = SoNDataset(im_paths_son_val,labels_son_val,['Average','Variance'],tr=composed)
dataset_son_test = SoNDataset(im_paths_son_test,labels_son_test,['Average','Variance'],tr=composed)

dataloader_son_train = DataLoader(dataset_son_train, batch_size=10,shuffle=True,num_workers=4,collate_fn=my_collate)
dataloader_son_val = DataLoader(dataset_son_val, batch_size=10,shuffle=False,num_workers=4,collate_fn=my_collate)
dataloader_son_test = DataLoader(dataset_son_test, batch_size=10,shuffle=False,num_workers=4,collate_fn=my_collate)

# Get the attribute names
temp = scipy.io.loadmat('attributes2.mat')
attr_names = [m[0][0] for m in temp['attributes']]
# get the labels and the image names
temp = scipy.io.loadmat('attributeLabels_continuous2.mat')
labels = temp['labels_cv']
temp = scipy.io.loadmat(SUN_dir+'trainval_idx.mat')
trainval_split = temp['sets']
temp = scipy.io.loadmat(SUN_dir+'images.mat')
im_names = [m[0][0] for m in temp['images']]
# Split in train-val-test (80-10-10)
train_indeces = np.where(trainval_split==0)[0].astype(int)
val_indeces = np.where(trainval_split==1)[0].astype(int)
test_indeces = np.where(trainval_split==2)[0].astype(int)
labels_train = labels[train_indeces,:]
im_names_train = [im_names[i] for i in train_indeces]
labels_val = labels[val_indeces,:]
im_names_val = [im_names[i] for i in val_indeces]
labels_test = labels[test_indeces,:]
im_names_test = [im_names[i] for i in test_indeces]


composed = transforms.Compose([Rescale((500,500)),Rotate(5),RandomCrop(450),VerticalFlip(),ToTensor()])

dataset_sun_train = SUNAttributesDataset(SUN_dir, im_names_train,labels_train,attr_names,tr=composed)
dataset_sun_val = SUNAttributesDataset(SUN_dir, im_names_val,labels_val,attr_names,tr=composed)
dataset_sun_test = SUNAttributesDataset(SUN_dir, im_names_test,labels_test,attr_names,tr=composed)

dataloader_sun_train = DataLoader(dataset_sun_train, batch_size=10,shuffle=True,num_workers=4)
dataloader_sun_val = DataLoader(dataset_sun_val, batch_size=10,shuffle=False,num_workers=4)
dataloader_sun_test = DataLoader(dataset_sun_test, batch_size=10,shuffle=False,num_workers=4)
#inspect_dataset(dataset,attr_names)


# Prepare initial model
basenet = models.resnet50(pretrained=True)
if freeze_basenet or freeze_sun_net:
    for param in basenet.parameters():
        param.requires_grad = False

if init_net_name is not None:
    # If the initial model has only been trained on SUN Attributes
    if init_net_name[0:8] == 'base_sun':
        net = nn.Sequential(*list(basenet.children())[:-2], NetSUNTop()).to(device)
        net.load_state_dict(torch.load(os.path.join(init_net_folder,init_net_name)))
        net = nn.Sequential(*list(net.children()), NetSoNTop(dataset_son_train.label_avg)).to(device)
    # If it has been trained both on SUN and SoN
    else:
        net = nn.Sequential(*list(basenet.children())[:-2], NetSUNTop(), NetSoNTop(dataset_son_train.label_avg)).to(device)
        net.load_state_dict(torch.load(os.path.join(init_net_folder,init_net_name)))
else:
    net = nn.Sequential(*list(basenet.children())[:-2], NetSUNTop(), NetSoNTop(dataset_son_train.label_avg)).to(device)


if freeze_basenet and not freeze_sun_net:
    optimizer = optim.Adam(net[-2:].parameters(),lr=init_lr,weight_decay=0.000)
elif freeze_sun_net:
    for param in net[-2].parameters():
        param.requires_grad = False
    optimizer = optim.Adam(list(net.children())[-1].parameters(), lr=init_lr, weight_decay=0.000)
else:
    optimizer = optim.Adam(net.parameters(),lr=init_lr,weight_decay=0.0000)

loss_sun = nn.BCELoss(reduction='none')
loss_son = nn.MSELoss(reduction='none')
nl = nn.Tanh()
epoch_loss_sun = []
epoch_loss_val_sun = []
epoch_loss_son = []
epoch_loss_val_son = []
epoch_loss_template = []


iter_num = 500
iter_num_val = 100
for epoch in range(n_epochs):  # loop over the dataset multiple times
    ############################################################################################
    # TRAIN
    ############################################################################################
    if np.any(np.array(reduce_lr) == epoch):
        optimizer.param_groups[-1]['lr'] = optimizer.param_groups[-1]['lr'] / 10
    iter_loss_sun = []
    iter_loss_son = []
    template_loss = 0
    dataloader_sun_train_for_epoch = iter(dataloader_sun_train)
    dataloader_son_train_for_epoch = iter(dataloader_son_train)

    net.train()
    done_with_sun = False
    done_with_son = False
    pbar = tqdm(total=iter_num)
    for i in range(iter_num):
        optimizer.zero_grad()
        # make FC layer non-negative
        net[-1].conv_templates_avg.weight.data.clamp_min_(0)
        net[-1].conv_templates_var.weight.data.clamp_min_(0)

        # SUN Attributes
        if not done_with_sun and (epoch>=epochs_only_son) and not freeze_sun_net:
            try:
                data = dataloader_sun_train_for_epoch.next()

            except:
                print('SUN exhausted in train')
                done_with_sun = True
                continue
            out_sun, _, maps, _ = net(data['image'].to(device))
            out_sun = loss_sun(nl(out_sun), data['label'].to(device))
            # out_sun = loss_sun(nl(out_sun), (data['label'] > 0.5).float().to(device))
            # out_sun = out_sun * ((data['label'] > 0.5) | (data['label'] < 0.1)).float().to(device)
            out_sun = out_sun.mean()
            iter_loss_sun.append(out_sun.item())

        # SoN
        if not done_with_son and (epoch>=epochs_only_sun):
            try:
                data = dataloader_son_train_for_epoch.next()
            except:
                print('SoN exhausted in train')
                done_with_son = True
                continue
            _, out_son, maps, _ = net(data['image'].to(device))
            out_son = loss_son(out_son, data['label'].to(device))
            out_son = out_son[:, 0].mean() + out_son[:, 1].mean() * 1
            iter_loss_son.append(out_son.item())
            template_loss = net[-1].conv_templates_avg.weight.abs().mean() + net[-1].conv_templates_var.weight.abs().mean()

        if freeze_sun_net or (epoch<epochs_only_son):
            out = out_son + 0.001*template_loss * out_son.max().detach() / template_loss.max().detach()
        elif (epoch<epochs_only_sun):
            out = out_sun
        else:
            out = out_sun + 0.1 * out_son * out_sun.max().detach() / out_son.max().detach() + 0.001 * template_loss * out_sun.max().detach() / template_loss.max().detach()

        if not (done_with_son and done_with_sun):
            out.backward()
            optimizer.step()


        pbar.set_description('Epoch ' + str(epoch) + '. Loss SUN:'+'%.4f'%(np.mean(iter_loss_sun)) + '. Loss SoN:'+'%.4f'%(np.mean(iter_loss_son))+ '. Loss template:'+'%.4f'%(1000*template_loss.item()))
        pbar.update()
    pbar.close()
    epoch_loss_sun.append(np.mean(iter_loss_sun))
    epoch_loss_son.append(np.mean(iter_loss_son))

    torch.save(net.state_dict(), os.path.join(net_folder, out_net_name))

    ############################################################################################
    # VAL
    ############################################################################################
    net.eval()
    torch.cuda.empty_cache()
    iter_loss_sun = []
    iter_loss_son = []
    dataloader_sun_val_for_epoch = iter(dataloader_sun_val)
    dataloader_son_val_for_epoch = iter(dataloader_son_val)
    done_with_sun = False
    done_with_son = False
    pbar = tqdm(total=iter_num_val)
    with torch.no_grad():
        for i in range(iter_num_val):
            # SUN Attributes
            if not done_with_sun:
                try:
                    data = dataloader_sun_val_for_epoch.next()
                except:
                    print('SUN exhausted in val')
                    done_with_sun = True
                    continue
                out_sun, _, maps, _ = net(data['image'].to(device))
                out_sun = loss_sun(nl(out_sun), data['label'].to(device))
                #out_sun = loss_sun(nl(out_sun), (data['label'] > 0.5).float().to(device))
                #out_sun = out_sun * ((data['label'] > 0.5) | (data['label'] < 0.1)).float().to(device)
                out_sun = out_sun.mean()
                iter_loss_sun.append(out_sun.item())

            # SoN
            if not done_with_son:
                try:
                    data = dataloader_son_val_for_epoch.next()
                except:
                    print('SoN exhausted in val')
                    done_with_son = True
                    continue
                _, out_son, maps, _ = net(data['image'].to(device))
                out_son = loss_son(out_son, data['label'].to(device))
                out_son = out_son[:, 0].mean() + out_son[:, 1].mean() * 1
                iter_loss_son.append(out_son.item())
            pbar.set_description('Val loss SUN: ' + '%.4f' % (np.mean(iter_loss_sun)) + '. Val loss SoN: ' + '%.4f' % (
            np.mean(iter_loss_son)))
            pbar.update()

    pbar.close()
    epoch_loss_val_sun.append(np.mean(iter_loss_sun))
    epoch_loss_val_son.append(np.mean(iter_loss_son))

