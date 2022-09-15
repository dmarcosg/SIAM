import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F

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

class NetSoNTopInterIndividual(nn.Module):
    def __init__(self,label_avg=np.array([0.0,0.0])):
        super(NetSoNTopInterIndividual, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(15)
        self.pool2avg = nn.AdaptiveAvgPool2d(1)
        self.pool2max = nn.AdaptiveMaxPool2d(1)

        self.conv_templates_avg = nn.Conv2d(33, 33 * 33, 15, groups=33, bias=False)
        self.conv_templates_var = nn.Conv2d(33, 33 * 33, 15, groups=33, bias=False)
        self.conv_templates_avg.weight.data.fill_(0.01)
        self.conv_templates_var.weight.data.fill_(0.01)

        self.fcp_avg = nn.Linear(33*33, 1)
        self.fcp_avg.weight.data.fill_(0.001)
        self.fcp_avg.bias.data.fill_(label_avg[0]/2)

        self.fcn_avg = nn.Linear(33 * 33, 1)
        self.fcn_avg.weight.data.fill_(-0.001)
        self.fcn_avg.bias.data.fill_(label_avg[0] / 2)

        self.fcp_var = nn.Linear(33*33, 1)
        self.fcp_var.weight.data.fill_(0.001)
        self.fcp_var.bias.data.fill_(label_avg[1] / 2)

        self.fcn_var = nn.Linear(33 * 33, 1)
        self.fcn_var.weight.data.fill_(0.001)
        self.fcn_var.bias.data.fill_(label_avg[1] / 2)



    def forward(self, maps):

        # pool to get attribute vector
        x_sun = (self.pool2avg(maps[:,0:33,:,:])).view(-1, 33)

        # match maps with map templates
        x_avg = self.conv_templates_avg(self.pool1(maps))
        x_avg = F.relu(x_avg)
        x_var = self.conv_templates_var(self.pool1(maps))
        x_var = F.relu(x_var)

        # get pairwise interactions
        x_avg = x_avg.squeeze()
        x_avg = self.pairwise_interaction(x_avg,x_avg)
        x_var = x_var.squeeze()
        x_var = self.pairwise_interaction(x_var,x_var)

        attr_contrib = [x_avg,x_var]
        #x_sort, used_attribs = x.sort(dim=1, descending=True)
        #x_sort[:, 12:-1] = 0
        #x = torch.gather(x_sort,1,torch.argsort(used_attribs,1))

        x_avg = self.fcp_avg(x_avg) + self.fcn_avg(x_avg)
        x_var = self.fcp_var(x_var) + self.fcn_var(x_var)

        x_son = torch.cat([x_avg, x_var], dim=1)

        return x_sun, x_son, maps, attr_contrib

    def pairwise_interaction(self,x,y):
        assert x.shape[1]==y.shape[1]
        s = np.sqrt(x.shape[1])
        assert s.is_integer()
        s = int(s)
        x = x.reshape(-1, s, s)
        y = y.reshape(-1, s, s)
        out = []
        for i in range(s):
            out.append(x[:,i,:]*y[:,:,i])
        out = torch.cat(out,dim=1)
        return out

class NetSoNTopInter(nn.Module):
    def __init__(self,label_avg=np.array([0.0,0.0])):
        super(NetSoNTopInter, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(15)
        self.pool2avg = nn.AdaptiveAvgPool2d(1)
        self.pool2max = nn.AdaptiveMaxPool2d(1)

        self.conv_templates_avg = nn.Conv2d(33, 33 * 1, 15, groups=33, bias=False)
        self.conv_templates_var = nn.Conv2d(33, 33 * 1, 15, groups=33, bias=False)
        self.conv_templates_avg.weight.data.fill_(0.01)
        self.conv_templates_var.weight.data.fill_(0.01)

        self.fcp_avg = nn.Bilinear(33,33,1)
        self.fcp_avg.weight.data.fill_(0.001)
        self.fcp_avg.bias.data.fill_(label_avg[0]/2)

        self.fcn_avg = nn.Bilinear(33,33,1)
        self.fcn_avg.weight.data.fill_(-0.001)
        self.fcn_avg.bias.data.fill_(label_avg[0] / 2)

        self.fcp_var = nn.Bilinear(33,33,1)
        self.fcp_var.weight.data.fill_(0.001)
        self.fcp_var.bias.data.fill_(label_avg[1] / 2)

        self.fcn_var = nn.Bilinear(33,33,1)
        self.fcn_var.weight.data.fill_(-0.001)
        self.fcn_var.bias.data.fill_(label_avg[1] / 2)



    def forward(self, maps):

        # pool to get attribute vector
        x_sun = (self.pool2avg(maps[:,0:33,:,:])).view(-1, 33)

        # match maps with map templates
        x_avg = self.conv_templates_avg(self.pool1(maps))
        x_avg = F.relu(x_avg)
        x_var = self.conv_templates_var(self.pool1(maps))
        x_var = F.relu(x_var)

        # get pairwise interactions
        x_avg = x_avg.squeeze(-1).squeeze(-1)
        x_var = x_var.squeeze(-1).squeeze(-1)

        attr_contrib = [((self.fcp_avg.weight+self.fcn_avg.weight) * x_avg.unsqueeze(-1) * x_avg.unsqueeze(1) ),
                        ((self.fcp_var.weight+self.fcn_var.weight) * x_var.unsqueeze(-1) * x_var.unsqueeze(1))]
        #x_sort, used_attribs = x.sort(dim=1, descending=True)
        #x_sort[:, 12:-1] = 0
        #x = torch.gather(x_sort,1,torch.argsort(used_attribs,1))

        x_avg = self.fcp_avg(x_avg,x_avg) + self.fcn_avg(x_avg,x_avg)
        x_var = self.fcp_var(x_var,x_var) + self.fcn_var(x_var,x_var)

        x_son = torch.cat([x_avg, x_var], dim=1)

        return x_sun, x_son, maps, attr_contrib