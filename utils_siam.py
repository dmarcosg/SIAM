from skimage import io, transform
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class SoNDataset(Dataset):
    """SoN scenicness dataset."""

    def __init__(self, im_paths,labels,attr_names, tr=None):
        self.im_paths = im_paths
        self.labels = labels
        self.label_avg = np.float32(labels.mean(axis=0))
        self.attr_names = attr_names
        self.transform = tr

    def __len__(self):
        return len(self.im_paths)


    def __getitem__(self, idx):
        img_name = self.im_paths[idx]
        try:
            image = io.imread(img_name)
        except:
            return None
        s = image.shape
        if len(s) > 3:
            print(img_name)
        if len(s) == 2:
            image = image[:, :, np.newaxis][:, :, [0, 0, 0]]
        if image.shape[2] != 3:
            image = image[:,:,0:3]
        image = np.ascontiguousarray(image)
        label = np.float32(self.labels[idx,:])
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)

class SUNAttributesDataset(Dataset):
    """SUN Attributes dataset."""

    def __init__(self, data_path,im_names,labels,attr_names, tr=None):
        self.data_path = data_path
        self.im_names = im_names
        self.labels = labels
        self.attr_names = attr_names
        self.transform = tr

    def __len__(self):
        return len(self.im_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_path, 'images',
                                self.im_names[idx])
        image = io.imread(img_name)
        s = image.shape
        if len(s) > 3:
            print(img_name)
        if len(s) == 2:
            image = image[:, :, np.newaxis][:, :, [0, 0, 0]]
        if image.shape[2] != 3:
            image = image[:,:,0:3]
        image = np.ascontiguousarray(image)
        label = np.float32(self.labels[idx,:])
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = np.transpose(image,(2, 0, 1))
        image = image.astype('float32') / 255

        return {'image': torch.from_numpy(image),
                'label': torch.from_numpy(label)}

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        done_resizing = False
        while not done_resizing:
            try:
                img = transform.resize(image, (new_h, new_w),mode='constant',anti_aliasing=True)
                done_resizing = True
            except:
                print('Issue resizing. Trying again.')


        return {'image': img, 'label': label}


class Rotate(object):
    """Rotate the image in a sample to a given size.

    Args:
        output_size (float): Maximum rotation.
    """

    def __init__(self, max_angle):
        assert isinstance(max_angle, (int, float))
        self.max_angle = max_angle

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        angle = (np.random.rand()-0.5)*2*self.max_angle
        done_rotating = False
        while not done_rotating:
            try:
                img = transform.rotate(image, angle)
                done_rotating = True
            except:
                print('Issue rotating. Trying again.')


        return {'image': img, 'label': label}

class VerticalFlip(object):
    """Flip the image in a sample with probability p.

    Args:
        p (float): Probability of vertical flip.
    """

    def __init__(self, p = 0.5):
        assert isinstance(p, (float))
        self.p = p

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if np.random.rand() < self.p:
            image = np.fliplr(image)


        return {'image': image, 'label': label}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        return {'image': image, 'label': label}

def inspect_dataset(dataset,attr_names):
    fig, axs = plt.subplots(nrows=1, ncols=2)
    for idx in range(20, 2000, 20):
        print(idx)
        show_image(dataset.__getitem__(idx), attr_names, fig, axs)
        axs[0].cla()
        axs[1].cla()

def show_image(sample,names,fig,axs):
    axs[0].imshow(sample['image'])
    label = sample['label']
    names = [names[i] for i in np.where(label>0)[0]]
    label = label[label>0]
    axs[1].barh(np.arange(len(label)),label)
    axs[1].set_yticks(np.arange(len(label)))
    axs[1].set_yticklabels(names)
    fig.tight_layout()
    plt.pause(0.0001)
    plt.waitforbuttonpress(timeout=-1)


class NetSUNSoNTopBaseMaps(nn.Module):
    def __init__(self,label_avg=[0,0]):
        super(NetSUNSoNTopBaseMaps, self).__init__()
        self.conv_sun = nn.Conv2d(2048, 33, 1)
        self.conv_son = nn.Conv2d(2048, 2, 1)
        self.pool_sun = nn.AdaptiveAvgPool2d(1)
        self.pool_son = nn.AdaptiveAvgPool2d(1)

        self.conv_son.bias.data[0].fill_(label_avg[0])
        self.conv_son.bias.data[1].fill_(label_avg[1])

    def forward(self, x):
        # get maps of attributes
        maps_sun = F.relu(self.conv_sun(x))
        maps_son = self.conv_son(x)

        # pool to get attribute vector
        x_sun = (self.pool_sun(maps_sun)).view(-1, 33)
        x_son = (self.pool_son(maps_son)).view(-1, 2)


        return x_sun, x_son, maps_sun, maps_son

class NetSUNSoNTopBase(nn.Module):
    def __init__(self,label_avg=[0,0]):
        super(NetSUNSoNTopBase, self).__init__()
        self.conv_sun = nn.Conv2d(2048, 34, 1)
        self.conv_son = nn.Conv2d(2048, 2, 1)

        self.pool = nn.AdaptiveAvgPool2d(1)

        #self.conv_son.bias.data[0].fill_(label_avg[0])
        #self.conv_son.bias.data[1].fill_(label_avg[1])

    def forward(self, x):
        # get maps of attributes
        x_sun = self.pool(F.relu(self.conv_sun(x))).squeeze()
        x_son = self.pool(self.conv_son(x)).squeeze()


        return x_sun, x_son

class NetSoNTopBase(nn.Module):
    def __init__(self,label_avg=[0,0]):
        super(NetSoNTopBase, self).__init__()
        self.conv_sun = nn.Conv2d(2048, 3, 1)
        self.conv_son = nn.Conv2d(2048, 2, 1)
        self.pool_sun = nn.AdaptiveAvgPool2d(1)
        self.pool_son = nn.AdaptiveAvgPool2d(1)

        self.conv_son.bias.data[0].fill_(label_avg[0])
        self.conv_son.bias.data[1].fill_(label_avg[1])

    def forward(self, x):
        # get maps of attributes
        maps_sun = F.relu(self.conv_sun(x))
        maps_son = self.conv_son(x)

        # pool to get attribute vector
        x_sun = (self.pool_sun(maps_sun)).view(-1, 33)
        x_son = (self.pool_son(maps_son)).view(-1, 2)


        return x_sun, x_son, maps_sun, maps_son


class NetSUNTop(nn.Module):
    def __init__(self):
        super(NetSUNTop, self).__init__()
        self.conv1 = nn.Conv2d(2048, 33, 1)


    def forward(self, x):
        # get maps of attributes
        maps = F.relu(self.conv1(x))

        maps[:, :, 0:2, :] = 0
        maps[:, :, -2:, :] = 0
        maps[:, :, :, 0:2] = 0
        maps[:, :, :, -2:] = 0

        return maps

class NetSoNTop(nn.Module):
    def __init__(self,label_avg=np.array([0.0,0.0])):
        super(NetSoNTop, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(15)
        self.pool2avg = nn.AdaptiveAvgPool2d(1)
        self.pool2max = nn.AdaptiveMaxPool2d(1)

        n_grops = 2
        self.conv_templates_avg = nn.Conv2d(33,33*n_grops,15,groups=33,bias=False)
        self.conv_templates_var = nn.Conv2d(33, 33 * n_grops, 15, groups=33, bias=False)
        self.conv_templates_avg.weight.data.fill_(0.01)
        self.conv_templates_var.weight.data.fill_(0.01)

        self.conv_combine_templates_avg = nn.Conv2d(33*n_grops,33,1,groups=33,bias=False)
        self.conv_combine_templates_var = nn.Conv2d(33 * n_grops, 33, 1, groups=33,bias=False)
        self.conv_combine_templates_avg.weight.data[:, 0, :, :].fill_(0.5)
        self.conv_combine_templates_avg.weight.data[:, 1, :, :].fill_(-0.5)
        self.conv_combine_templates_var.weight.data[:, 0, :, :].fill_(0.5)
        self.conv_combine_templates_var.weight.data[:, 1, :, :].fill_(-0.5)

        self.fc1_avg = nn.Linear(33, 1)
        self.fc1_var = nn.Linear(33, 1)

        self.fc1_avg.weight.data.fill_(1/33)
        self.fc1_var.weight.data.fill_(1/33)

        self.fc1_avg.bias.data.fill_(label_avg[0])
        self.fc1_var.bias.data.fill_(label_avg[1])

    def forward(self, maps):

        # pool to get attribute vector
        x_sun = (self.pool2avg(maps[:,0:33,:,:])).view(-1, 33)

        # match maps with map templates
        x_avg = self.conv_templates_avg(self.pool1(maps))
        x_avg = F.relu(x_avg)
        x_var = self.conv_templates_var(self.pool1(maps))
        x_var = F.relu(x_var)

        # combine all templates corresponding to each attribute
        x_avg = self.conv_combine_templates_avg(x_avg)
        x_var = self.conv_combine_templates_var(x_var)
        #x_avg = self.bn_combine(x_avg)
        #x_var = self.bn_combine(x_var)



        x_avg = x_avg.view(-1, 33)
        x_var = x_var.view(-1, 33)

        attr_contrib = [x_avg,x_var]
        #x_sort, used_attribs = x.sort(dim=1, descending=True)
        #x_sort[:, 12:-1] = 0
        #x = torch.gather(x_sort,1,torch.argsort(used_attribs,1))

        x_avg = self.fc1_avg(x_avg)
        x_var = self.fc1_var(x_var)

        x_son = torch.cat([x_avg, x_var], dim=1)

        return x_sun, x_son, maps, attr_contrib




