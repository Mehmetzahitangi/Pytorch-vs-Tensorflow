import torch
from torch import nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self,in_channels,out_channels, downsample = False):
        super(Block,self).__init__()

        self.downsample = downsample

        if downsample:
            self.basic_conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=2,padding=1)
            self.downsample_conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels, kernel_size=1, stride=2,padding=0)
        else:
            self.basic_conv = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1)

        self.basic_batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.bootleneck_conv = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1)
        self.bootleneck_batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self,x):
        residual = x

        out = self.basic_conv(x)
        out = self.basic_batch_norm(out)
        out = self.relu(out)

        out = self.bootleneck_conv(out)
        out = F.dropout(out, 0.5)
        out = self.bootleneck_batch_norm(out)
        out = self.relu(out)

        if self.downsample:
            residual = self.downsample_conv(residual)

        # print("out: {} and residual: {}".format(out.shape, residual.shape))
        out = out.clone() + residual
        # print("out: {} and residual: {} after summation".format(out.shape, residual.shape))
        return out



class ResNet18(nn.Module):
    def __init__(self,num_classes):
        super().__init__()


        self.conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7, stride=2, padding=3)
        self.batch_norm1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = Block(64,64)
        self.layer2 = Block(64,64)
        self.layer3 = Block(64,128, downsample = True)
        self.layer4 = Block(128,128)
        self.layer5 = Block(128,256, downsample = True)
        self.layer6 = Block(256,256)
        self.layer7 = Block(256,512, downsample = True)
        self.layer8 = Block(512,512)

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fully_connected = nn.Linear(512, num_classes)
        #self.fully_connected = nn.Linear(512, 1) # binary classification

    def forward(self,x):
        out = self.conv1(x)
        out = self.batch_norm1(out)
        out = self.relu(out)

        #print("Bottleneck Part Intro conv 7*7: ", out.shape)
        out = self.maxpool1(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)

        out = self.avg_pool(out)
        out = self.fully_connected(out.view(out.shape[0],-1)) # out.shape[0]  get batch size,  -1 =>>  layers will be flat except batchsize_layer

        return out

#model = ResNet18(num_classes=2)
#x = torch.rand(size=(1,3,224,224))

#out = model(x)
#print("Last output shape: ",out.shape)