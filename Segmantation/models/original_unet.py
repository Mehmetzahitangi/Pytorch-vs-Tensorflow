import torch
import torch.nn as nn
import torch.nn.functional as F

def double_down_conv(in_channel,out_channel):
    down_conv = nn.Sequential(
        nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=3),
        nn.ReLU(),
        nn.Conv2d(in_channels=out_channel,out_channels=out_channel,kernel_size=3),
        nn.ReLU()
    )

    return down_conv


def double_up_conv(in_channel,out_channel):
    up_conv = nn.Sequential(
        nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=3),
        nn.ReLU(),
        nn.Conv2d(in_channels=out_channel,out_channels=out_channel,kernel_size=3),
        nn.ReLU()
    )
    return up_conv

def crop_image(tensor,target_tensor):
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    delta = tensor_size - target_size #target_size - tensor_size
    delta = delta // 2
    return tensor[:,:,delta:tensor_size-delta,delta:tensor_size-delta]



class UNet(nn.Module):
    def __init__(self):
        super(UNet,self).__init__()
        
        # encoder
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.down_conv1 = double_down_conv(1,64)
        self.down_conv2 = double_down_conv(64,128)
        self.down_conv3 = double_down_conv(128,256)
        self.down_conv4 = double_down_conv(256,512)
        self.down_conv5 = double_down_conv(512,1024)

        # decoder
        self.up_conv1 = nn.ConvTranspose2d(in_channels=1024,out_channels=512,kernel_size=2,stride=2)
        self.up_conv1_3x3 = double_up_conv(in_channel=1024,out_channel=512)

        self.up_conv2 = nn.ConvTranspose2d(in_channels=512,out_channels=256,kernel_size=2,stride=2)
        self.up_conv2_3x3 = double_up_conv(in_channel=512,out_channel=256)

        self.up_conv3 = nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=2,stride=2)
        self.up_conv3_3x3 = double_up_conv(in_channel=256,out_channel=128)

        self.up_conv4 = nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=2,stride=2)
        self.up_conv4_3x3 = double_up_conv(in_channel=128,out_channel=64)

        self.out = nn.Conv2d(in_channels=64,out_channels=2,kernel_size=1,padding=0) # original paper


    def forward(self,image):
        x1 = self.down_conv1(image) # 1,64,568,568
        #print("x1: ",x1.shape)
        x2 = self.max_pool_2x2(x1) # 1,64,284,284
        #print("x2: ",x2.shape)

        x3 = self.down_conv2(x2) # 1,128,280,280
        #print("x3: ",x3.shape)
        x4 = self.max_pool_2x2(x3) # 1,128,140,140
        #print("x4: ",x4.shape)

        x5 = self.down_conv3(x4) # 1,256,136,136
       #print("x5: ",x5.shape)
        x6 = self.max_pool_2x2(x5) # 1,256,68,68
        #print("x6: ",x6.shape)

        x7 = self.down_conv4(x6) # 1,512,64,64
        #print("x7: ",x7.shape)
        x8 = self.max_pool_2x2(x7) # 1,512,32,32
        #print("x8: ",x8.shape)

        x9 = self.down_conv5(x8) # 1,1024,28,28
        #print("x9: ",x9.shape)

        # decoder
        x10 = self.up_conv1(x9) # 1,512,56,56
        #print("X10: ",x10.shape)
        x11 = torch.cat((x10,crop_image(target_tensor=x10,tensor=x7)),dim=1)
        print("X11 shape: ",x11.shape)
        x12 = self.up_conv1_3x3(x11)
        print("X12 shape: ",x12.shape)
        
        x13 = self.up_conv2(x12) # 1,512,52,52
        print("x13: ",x13.shape)
        x14 = torch.cat((x13,crop_image(tensor=x5,target_tensor= x13)),dim=1) # 1,512,104,104
        #x14 = torch.cat((x13,x5),dim=1)
        #print("x14: ",x14.shape)
        x15 = self.up_conv2_3x3(x14) # 1,256,100,100
        #print("x15: ",x15.shape)


        x16 = self.up_conv3(x15) # 1,128,100,100
        #print("x16: ",x16.shape)
        #x17 = torch.cat((x16,x3),dim=1)
        x17 = torch.cat((x16,crop_image(tensor=x3,target_tensor = x16)),dim=1) # 1,256,200,200
        #print("x17: ",x17.shape)
        x18 = self.up_conv3_3x3(x17) # 1,128,196,196
        #print("x18: ",x18.shape)


        x19 = self.up_conv4(x18) # 1,64,196,196
        #print("x19: ",x19.shape)
        #x20 = torch.cat((x19,x1),dim=1)
        x20 = torch.cat((x19,crop_image(tensor = x1, target_tensor = x19)),dim=1) # 1,128,392,392
        #print("x20: ",x20.shape)
        x21 = self.up_conv4_3x3(x20) # 1,64,388,388
        #print("x21: ",x21.shape)

        out = self.out(x21)
        out = F.log_softmax(out, 1)
        #print("out: ",out.shape)

        return out



#model = UNet()

#x = torch.rand(size=(1,1,572,572))

#output = model(x)