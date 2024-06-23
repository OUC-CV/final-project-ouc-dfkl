import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import math
import numpy as np


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class LayerActivation:
    features = None

    def __init__(self, model, layer_num):
        self.hook = model[layer_num].register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output.cpu()

    def remove(self):
        self.hook.remove()


class Encoder(nn.Module):
    def __init__(self, device):
        super(Encoder, self).__init__()
        self.vgg = models.vgg16(pretrained=True)
        self.device = device
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.skip1 = LayerActivation(self.vgg.features, 3)
        self.skip2 = LayerActivation(self.vgg.features, 8)
        self.skip3 = LayerActivation(self.vgg.features, 15)
        self.skip4 = LayerActivation(self.vgg.features, 22)
        self.skip5 = LayerActivation(self.vgg.features, 29)

    def forward(self, x):
        self.vgg(x)

        return x, self.skip1.features.to(self.device), self.skip2.features.to(self.device) \
            , self.skip3.features.to(self.device), self.skip4.features.to(self.device), self.skip5.features.to(self.device),


def upsample(x, convT, skip, conv1x1, device):
    x = convT(x)
    bn = nn.BatchNorm2d(x.shape[1]).to(device)
    x = bn(x)
    x = F.leaky_relu(x, 0.2)

    skip = torch.log(skip ** 2 + 1.0/255.0)
    x = torch.cat([x, skip], dim=1)
    x = conv1x1(x)
    return x


def upsample_last(x, conv1x1_64_3, skip, conv1x1_6_3, device):
    x = conv1x1_64_3(x)
    bn = nn.BatchNorm2d(x.shape[1]).to(device)
    x = bn(x)
    x = F.leaky_relu(x, 0.2)

    skip = torch.log(skip ** 2 + 1.0 / 255.0)
    x = torch.cat([x, skip], dim=1)
    x = conv1x1_6_3(x)
    return x


class Decoder(nn.Module):
    def __init__(self, device):
        super(Decoder, self).__init__()
        self.device = device
        self.latent_representation = nn.Sequential(
            Conv2d(512, 512, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            Conv2d(512, 512, kernel_size=3, padding=1)
        )
        self.convTranspose_5 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.conv1x1_5 = Conv2d(1024, 512, kernel_size=1)

        self.convTranspose_4 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.conv1x1_4 = Conv2d(1024, 512, kernel_size=1)

        self.convTranspose_3 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.conv1x1_3 = Conv2d(512, 256, kernel_size=1)

        self.convTranspose_2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.conv1x1_2 = Conv2d(256, 128, kernel_size=1)

        self.convTranspose_1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.conv1x1_1 = Conv2d(128, 64, kernel_size=1)

        self.conv1x1_64_3 = Conv2d(64, 3, kernel_size=1)
        self.conv1x1_6_3 = Conv2d(6, 3, kernel_size=1)

    def forward(self, skip0, skip1, skip2, skip3, skip4, skip5):
        x = self.latent_representation(skip5)
        x = upsample(x, self.convTranspose_5, skip5, self.conv1x1_5, self.device)
        x = upsample(x, self.convTranspose_4, skip4, self.conv1x1_4, self.device)
        x = upsample(x, self.convTranspose_3, skip3, self.conv1x1_3, self.device)
        x = upsample(x, self.convTranspose_2, skip2, self.conv1x1_2, self.device)
        x = upsample(x, self.convTranspose_1, skip1, self.conv1x1_1, self.device)
        x = upsample_last(x, self.conv1x1_64_3, skip0, self.conv1x1_6_3, self.device)
        return x

   
class enhance_net_nopool(nn.Module):
    
    def __init__(self):
        super(enhance_net_nopool, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        
        number_f = 32
        self.e_conv1 = nn.Conv2d(3,number_f,3,1,1,bias=True) 
        self.e_conv2 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
        self.e_conv3 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
        self.e_conv4 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
        self.e_conv5 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
        self.e_conv6 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
        self.e_conv7 = nn.Conv2d(number_f*2,24,3,1,1,bias=True) 

        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)


		
    def forward(self, x):

        x1 = self.relu(self.e_conv1(x))
		# p1 = self.maxpool(x1)
        x2 = self.relu(self.e_conv2(x1))
		# p2 = self.maxpool(x2)
        x3 = self.relu(self.e_conv3(x2))
		# p3 = self.maxpool(x3)
        x4 = self.relu(self.e_conv4(x3))

        x5 = self.relu(self.e_conv5(torch.cat([x3,x4],1)))
		# x5 = self.upsample(x5)
        x6 = self.relu(self.e_conv6(torch.cat([x2,x5],1)))

        x_r = F.tanh(self.e_conv7(torch.cat([x1,x6],1)))
        r1,r2,r3,r4,r5,r6,r7,r8 = torch.split(x_r, 3, dim=1)


        x = x + r1*(torch.pow(x,2)-x)
        x = x + r2*(torch.pow(x,2)-x)
        x = x + r3*(torch.pow(x,2)-x)
        enhance_image_1 = x + r4*(torch.pow(x,2)-x)		
        x = enhance_image_1 + r5*(torch.pow(enhance_image_1,2)-enhance_image_1)		
        x = x + r6*(torch.pow(x,2)-x)	
        x = x + r7*(torch.pow(x,2)-x)
        x=x*0.6
        enhance_image = x + r8*(torch.pow(x,2)-x)
        r = torch.cat([r1,r2,r3,r4,r5,r6,r7,r8],1)

        return enhance_image_1,enhance_image,r
###降噪
class DnCNN(nn.Module):
    def __init__(self, channels=3, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
    def forward(self, x):
        out = self.dncnn(x)
        return out
    
class SRCNN(nn.Module):
    def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        x = F.interpolate(x, size=(768, 1023), mode='bicubic', align_corners=False)
        return x

class Model(nn.Module):
    def __init__(self, device):
        super(Model, self).__init__()
        self.encoder = Encoder(device)
        self.decoder = Decoder(device)

    def forward(self, x):
        x = x.float()
        skip0, skip1, skip2, skip3, skip4, skip5 = self.encoder(x)
        x = self.decoder(skip0, skip1, skip2, skip3, skip4, skip5)
        return x

