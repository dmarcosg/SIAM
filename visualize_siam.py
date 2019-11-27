import scipy.io
from scipy.misc import imsave
from skimage import transform
import os
import csv
import numpy as np
from torchvision import transforms, utils
import torchvision.models as models
from utils_siam import SUNAttributesDataset, SoNDataset, show_image, inspect_dataset, ToTensor, Rescale, RandomCrop, VerticalFlip, NetSUNSoNTopBase,NetSUNTop, NetSoNTop
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from tqdm import tqdm
import visdom
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap



gpu_no = 0  # Set to False for cpu-version
net_folder = 'models'
net_name = 'sunson_siam.pt'
net_folder2 = 'models'
net_name2 = 'son_baseline.pt'
only_most_active = True

# SoN dataset
data_path_son = '[...]/Datasets/SoN/'
image_folders = ['images1','images2','images3','images4','images5','images6']
im_paths = []
son_avg = []
son_var = []
with open(os.path.join(data_path_son,'SoN_Votes.csv'), 'r') as csvfile:
     SoN_reader = csv.reader(csvfile, delimiter=',')
     for row in SoN_reader:
         for image_folder in image_folders:
             im_path = os.path.join(data_path_son,image_folder,row[0]+'.jpg')
             if os.path.isfile(im_path):
                im_paths.append(im_path)
                son_avg.append(np.float32(row[3]))
                son_var.append(np.float32(row[4]))

labels_son = np.array([son_avg,son_var]).transpose()

im_paths_son_train = im_paths[0:150000]
labels_son_train = labels_son[0:150000,:]
im_paths_son_val = im_paths[150000:155000]
labels_son_val = labels_son[150000:155000,:]
im_paths_son_test = im_paths[160000:190000]
labels_son_test = labels_son[160000:190000,:]

composed = transforms.Compose([Rescale(500),ToTensor()])
dataset_son_train = SoNDataset(im_paths_son_train,labels_son_train,['Average','Variance'],tr=composed)
dataset_son_val = SoNDataset(im_paths_son_val,labels_son_val,['Average','Variance'],tr=composed)
dataset_son_test = SoNDataset(im_paths_son_test,labels_son_test,['Average','Variance'],tr=composed)

dataloader_son_train = DataLoader(dataset_son_train, batch_size=1,shuffle=False,num_workers=1)
dataloader_son_val = DataLoader(dataset_son_val, batch_size=1,shuffle=False,num_workers=1)
dataloader_son_test = DataLoader(dataset_son_test, batch_size=1,shuffle=False,num_workers=1)


# SUN Attributes
data_path = '[...]/Datasets/SUN/SUNAttributeDB/'
# Get the attribute names
temp = scipy.io.loadmat(data_path+'attributes2.mat')
attr_names = [m[0][0] for m in temp['attributes']]
# get the labels and the image names
temp = scipy.io.loadmat(data_path+'attributeLabels_continuous2.mat')
labels = temp['labels_cv']
temp = scipy.io.loadmat(data_path+'trainval_idx.mat')
trainval_split = temp['sets']
temp = scipy.io.loadmat(data_path+'images.mat')
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


if type(gpu_no) == int:
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

composed = transforms.Compose([Rescale(500),ToTensor()])

dataset_sun_train = SUNAttributesDataset(data_path,im_names_train,labels_train,attr_names,tr=composed)
dataset_sun_val = SUNAttributesDataset(data_path,im_names_val,labels_val,attr_names,tr=composed)
dataset_sun_test = SUNAttributesDataset(data_path,im_names_test,labels_test,attr_names,tr=composed)

dataloader_sun_train = DataLoader(dataset_sun_train, batch_size=1,shuffle=False,num_workers=1)
dataloader_sun_val = DataLoader(dataset_sun_val, batch_size=1,shuffle=False,num_workers=1)
dataloader_sun_test = DataLoader(dataset_sun_test, batch_size=1,shuffle=False,num_workers=1)
#inspect_dataset(dataset,attr_names)



basenet = models.resnet50(pretrained=True)
basenet2 = models.resnet50(pretrained=True)




#net(tr(sample)['image'].unsqueeze(0))


# Our model
net = nn.Sequential(*list(basenet.children())[:-2], NetSUNTop(), NetSoNTop(dataset_son_train.label_avg)).to(device)
net.load_state_dict(torch.load(os.path.join(net_folder,net_name)))
for param in net.parameters():
    param.requires_grad = False
net.eval()

#Baseline
net2 = nn.Sequential(*list(basenet2.children())[:-1],NetSUNSoNTopBase()).to(device)
net2.load_state_dict(torch.load(os.path.join(net_folder2,net_name2)))
net2.eval()
basenet2 = nn.Sequential(*list(net2.children())[:-2]).to(device)
contrib = net2[-1].conv_son.weight.data[0,:].squeeze().cpu().numpy()
for param in net2.parameters():
    param.requires_grad = False
net2.eval()

# Get tamplates
all_templates = net[-1].conv_templates_avg.weight.data.cpu().numpy()
map_contrib_avg = net[-1].conv_combine_templates_avg.weight.data.cpu().numpy()
weigths_avg = net[-1].fc1_avg.weight.data.cpu().numpy()[0]
templates = np.zeros((len(attr_names),all_templates.shape[2],all_templates.shape[3]))
for i in range(len(attr_names)):
    map_idx = np.arange(i*2,(i+1)*2)
    this_contrib = map_contrib_avg[i,:,0,0] * weigths_avg[i]
    template = all_templates[map_idx[0],0,:,:]*this_contrib[0] + all_templates[map_idx[1],0,:,:]*this_contrib[1]
    templates[i,:,:] = template


#vis = visdom.Visdom()
nrows = 2
if only_most_active:
    ncols=8
else:
    ncols = 15
#fig, axes = plt.subplots(nrows=11, ncols=20)
#fig.subplots_adjust(hspace=0.5)
fig=plt.figure(figsize=(12,2),constrained_layout=False)
gs = fig.add_gridspec(nrows=nrows, ncols=ncols, hspace=0.5)
pool_avg = nn.AdaptiveAvgPool2d(1)
pool_avg_15 = nn.AdaptiveAvgPool2d(15)
dataset = dataset_son_test
iter_num = 2000

colors = [(1, 0, 0.5), (1, 1, 1), (0, 0.5, 0)]
cmap=LinearSegmentedColormap.from_list('rg',colors, N=256)
plt.register_cmap(cmap=cmap)
over = [29714, 29234, 23436, 25512, 15542, 22496, 16544,  6301, 19823,23196, 25747, 24913, 15885, 13164, 18695,  5973, 20092, 21161, 26815, 28571]
under = [ 6184, 14914, 20894,  5644,  6308, 10608,  9823, 12829,  4747, 9134, 12527, 27883, 12319, 24707, 16316,  3199,  3292,  8681, 19292, 20234]
for i in under:#range(iter_num):

    data = dataset[i]
    data = dataloader_son_val.collate_fn([data])
    #Baseline
    out_sun2, out_son2 = net2(data['image'].to(device))
    maps2 = basenet2(data['image'].to(device))
    maps2 = maps2.squeeze().cpu().numpy()
    map2 = np.zeros(maps2.shape[1:])
    for m in range(maps2.shape[0]):
        map2 += maps2[m, :, :] * contrib[m]

    map2[ 0:2, :] = 0
    map2[ -2:, :] = 0
    map2[ :, 0:2] = 0
    map2[ :, -2:] = 0
    # Ours
    out_sun, out_son, maps, attr_contrib = net(data['image'].to(device))
    map = np.zeros(maps.shape[2:])
    for m in range(maps.shape[1]):
        map += maps[0, m, :, :].cpu().numpy() * transform.resize(templates[m,:,:],map.shape)
    maps=pool_avg_15(maps)
    ax_im = fig.add_subplot(gs[:,0:nrows])
    #vis.image(data['image'].squeeze() * 255,win=0)
    ax_im.imshow(data['image'].squeeze().permute(1, 2, 0) * 255)
    #imsave('image_example/im.png',data['image'].squeeze().permute(1, 2, 0) * 255)
    ax_im.set_xticks([])
    ax_im.set_yticks([])
    if (data['label'].shape[1] == 2):
        ax_im.set_title('Crowd scenicness: '+'%.2f' % data['label'][0,0], family='DejaVu Serif')
    else:
        ax_im.set_title('Pred: ' + '%.2f' % out_son[0, 0].item(), family='DejaVu Serif')

    ax_map = fig.add_subplot(gs[0:int(nrows/2), nrows:nrows*2])
    ax_map.imshow(map,clim=[-0.05,0.05],cmap='rg')
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    ax_map.set_title('Our model: ' + '%.2f' % out_son[0, 0].item(), family='DejaVu Serif')

    ax_map2 = fig.add_subplot(gs[int(nrows/2):, nrows:nrows * 2])
    ax_map2.imshow(map2, clim=[-5, 5], cmap='rg')
    ax_map2.set_xticks([])
    ax_map2.set_yticks([])
    ax_map2.set_title('Baseline: ' + '%.2f' % out_son2[0].item(), family='DejaVu Serif')

    map_list = []
    max_val = 4#maps.max().item()
    attr_contrib = attr_contrib[0].detach().cpu().numpy()[0,:]
    attr_weights = net[-1].fc1_avg.weight.data.detach().cpu().numpy()[0,:]
    attr_weights *= attr_contrib
    if only_most_active:
        order = np.argsort(-np.abs(attr_weights))
    else:
        order = np.arange(33)
    #attr_weights /= np.abs(attr_weights).max()
    for o in range(nrows*(ncols-nrows*2)):
        j = order[o]
        map_list.append(maps[0, j, :, :].detach().cpu().numpy())
        pos = np.unravel_index(o,(nrows,ncols-nrows*2))
        sub = fig.add_subplot(gs[pos[0],pos[1]+nrows*2])
        sub.imshow(maps[0, j, :, :].detach().cpu().numpy(),vmax=max_val,vmin=0,cmap='gray')

        #imsave('image_example/'+attr_names[j]+'.png', maps[0, j, :, :].detach().cpu().numpy() * 50)

        #sub.set_title(attr_names[j],fontsize=7,y=0.85+(j%2)*0.2)
        title = attr_names[j]
        title = title.split('_')[0]
        if data['label'].shape[1] > 2:
            title += ' (%.1f'%(data['label'][0,j])+')'
        sub.set_title(title, fontsize=11, y=1.0, family='DejaVu Serif')
        sub.set_xticks([])
        sub.set_yticks([])
        for s in sub.spines:
            sub.spines[s].set_linewidth(np.minimum(8,32 * np.abs(attr_weights[j])) )
            if attr_weights[j] > 0:
                sub.spines[s].set_color((0, 0.5, 0))
            else:
                sub.spines[s].set_color((1, 0, 0.5))

    #fig.savefig('figures/ours_underestimates/Figure_'+str(i)+'.svg', bbox_inches='tight',transparent=True)
    fig.show()
    plt.pause(0.0001)
    plt.waitforbuttonpress(timeout=-1)


