import torch.nn as nn
import torch
from torchvision import models

import torch.nn.functional as F
from misc.utils import *

import pdb

# model_path = '../PyTorch_Pretrained/resnet101-5d3b4d8f.pth'

class Res50_SCAR(nn.Module):
    def __init__(self, pretrained=True):
        super(Res50_SCAR, self).__init__()
        self.seen = 0
        self.backend_feat  = [512, 512, 512,256,128,64]
        self.frontend = []
        
        self.backend = make_layers(self.backend_feat,in_channels = 1024,dilation = True)
        self.output_layer = SCAModule(64, 1)

        # self.output_layer = nn.Sequential(nn.Conv2d(64, 1, kernel_size=1),nn.ReLU())

        initialize_weights(self.modules())

        res = models.resnet50(pretrained=pretrained)

        self.frontend = nn.Sequential(
            res.conv1, res.bn1, res.relu, res.maxpool, res.layer1, res.layer2
        )
        self.own_reslayer_3 = make_res_layer(Bottleneck, 256, 6, stride=1)        
        self.own_reslayer_3.load_state_dict(res.layer3.state_dict())

        


    def forward(self,x):
        x = self.frontend(x)

        x = self.own_reslayer_3(x)

        # pdb.set_trace()
        x = self.backend(x)

        x = self.output_layer(x)

        x = F.interpolate(x,scale_factor=8, mode='nearest')
        return x
            
                
def make_layers(cfg, in_channels = 3,batch_norm=False,dilation = False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)      


def make_res_layer(block, planes, blocks, stride=1):

    downsample = None
    inplanes=512
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * block.expansion,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )

    layers = []
    layers.append(block(inplanes, planes, stride, downsample))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(inplanes, planes))

    return nn.Sequential(*layers)  


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out        

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, NL='relu', same_padding=False, bn=True, bias=True):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) // 2) if same_padding else 0

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias)

        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        if NL == 'relu' :
            self.relu = nn.ReLU(inplace=True) 
        elif NL == 'prelu':
            self.relu = nn.PReLU() 
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class SCAModule(nn.Module):
    def __init__(self, inn, out):
        super(SCAModule, self).__init__()
        base = inn // 4
        self.conv_sa = nn.Sequential(Conv2d(inn, base, 3, same_padding=True, bias=False),
                                     SAM(base),
                                     Conv2d(base, base, 3, same_padding=True, bias=False)
                                     )       
        self.conv_ca = nn.Sequential(Conv2d(inn, base, 3, same_padding=True, bias=False),
                                     CAM(base),
                                     Conv2d(base, base, 3, same_padding=True, bias=False)
                                     )
        self.conv_cat = Conv2d(base*2, out, 1, same_padding=True, bn=False)

    def forward(self, x):
        sa_feat = self.conv_sa(x)
        ca_feat = self.conv_ca(x)
        cat_feat = torch.cat((sa_feat,ca_feat),1)
        cat_feat = self.conv_cat(cat_feat)
        return cat_feat   


class SAM(nn.Module):
    def __init__(self, channel):
        super(SAM, self).__init__()
        self.para_lambda = nn.Parameter(torch.zeros(1))
        self.query_conv = Conv2d(channel, channel//8, 1, NL='none')
        self.key_conv = Conv2d(channel, channel//8, 1, NL='none')
        self.value_conv = Conv2d(channel, channel, 1, NL='none')

    def forward(self, x):
        N, C, H, W = x.size() 
        proj_query = self.query_conv(x).view(N, -1, W*H).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(N, -1, W*H)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy,dim=-1)
        proj_value = self.value_conv(x).view(N, -1, W*H)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(N, C, H, W)

        out = self.para_lambda*out + x
        return out

class CAM(nn.Module):
    def __init__(self, in_dim):
        super(CAM, self).__init__()
        self.para_mu = nn.Parameter(torch.zeros(1))

    def forward(self,x):
        N, C, H, W = x.size() 
        proj_query = x.view(N, C, -1)
        proj_key = x.view(N, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = F.softmax(energy,dim=-1)
        proj_value = x.view(N, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(N, C, H, W)

        out = self.para_mu*out + x
        return out