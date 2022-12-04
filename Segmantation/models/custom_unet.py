import torch
import torch.nn as nn
import torch.nn.functional as F

# input image and output image are the same size
#  
def double_down_conv(in_channel, out_channel):
    down_conv = nn.Sequential(
        nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1),  # org padding = 0
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1),  # org padding = 0
        nn.ReLU(inplace=True),
    )
    return down_conv


def double_up_conv(in_channel, out_channel):
    up_conv = nn.Sequential(
        nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1),  # org padding = 0
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1),  # org padding = 0
        nn.ReLU(inplace=True),
    )
    return up_conv




class UNet_Custom(nn.Module):
    def __init__(self, num_classes):
        super(UNet_Custom, self).__init__()
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv1 = double_down_conv(in_channel=1, out_channel=64)
        self.down_conv2 = double_down_conv(in_channel=64, out_channel=128)
        self.down_conv3 = double_down_conv(in_channel=128, out_channel=256)
        self.down_conv4 = double_down_conv(in_channel=256, out_channel=512)
        self.down_conv5 = double_down_conv(in_channel=512, out_channel=1024)

        self.up_conv1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.up_conv1_3x3 = double_up_conv(in_channel=1024, out_channel=512)

        self.up_conv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.up_conv2_3x3 = double_up_conv(in_channel=512, out_channel=256)

        self.up_conv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.up_conv3_3x3 = double_up_conv(in_channel=256, out_channel=128)

        self.up_conv4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.up_conv4_3x3 = double_up_conv(in_channel=128, out_channel=64)

        self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=3, padding=1)

    def forward(self, image):
        # encoder
        x1 = self.down_conv1(image)  
        # print("x1: ",x1.shape)
        x2 = self.max_pool_2x2(x1) 
        # print("x2: ",x2.shape)

        x3 = self.down_conv2(x2)  
        # print("x3: ",x3.shape)
        x4 = self.max_pool_2x2(x3) 
        # print("x4: ",x4.shape)

        x5 = self.down_conv3(x4) 
        # print("x5: ",x5.shape)
        x6 = self.max_pool_2x2(x5)  
        # print("x6: ",x6.shape)

        x7 = self.down_conv4(x6)  
        # print("x7: ",x7.shape)
        x8 = self.max_pool_2x2(x7)  
        # print("x8: ",x8.shape)

        x9 = self.down_conv5(x8)  
        # print("x9: ",x9.shape)

        # decoder
        x10 = self.up_conv1(x9)  
        # print("x10: ",x10.shape)
        x11 = torch.cat((x10, x7), dim=1)
        # print("x11: ",x11.shape)
        x12 = self.up_conv1_3x3(x11)  
        # print("x12: ",x12.shape)

        x13 = self.up_conv2(x12) 
        # print("x13: ",x13.shape)
        x14 = torch.cat((x13, x5), dim=1)
        # print("x14: ",x14.shape)
        x15 = self.up_conv2_3x3(x14) 
        # print("x15: ",x15.shape)

        x16 = self.up_conv3(x15) 
        # print("x16: ",x16.shape)
        x17 = torch.cat((x16, x3), dim=1)
        # print("x17: ",x17.shape)
        x18 = self.up_conv3_3x3(x17) 
        # print("x18: ",x18.shape)

        x19 = self.up_conv4(x18)  
        # print("x19: ",x19.shape)
        x20 = torch.cat((x19, x1), dim=1)
        # print("x20: ",x20.shape)
        x21 = self.up_conv4_3x3(x20)  
        # print("x21: ",x21.shape)

        out = self.out(x21)
        out = F.log_softmax(out, 1)

        return out


# """DEBUG"""
model = UNet_Custom(3)
x = torch.rand(1,1,128,256)
out = model(x)
print("Out shape: ",out.shape)