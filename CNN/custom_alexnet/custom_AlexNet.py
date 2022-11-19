import torch 
import torch.nn as nn 

class AlexNet(nn.Module): #inherit from pytorch
    def __init__(self,num_classes):
        super(AlexNet,self).__init__()

        # self.relu = nn.ReLU()

        # self.conv1 = nn.Conv2d(in_channels=3,out_channels=96, kernel_size=11, stride=4 ,padding=0)
        # self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        # self.conv2 = nn.Conv2d(in_channels=96,out_channels=256,kernel_size=5, stride=1, padding=2)
        # self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        # self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        # self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        # self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        # self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        # self.adaptivepool = nn.AdaptiveAvgPool2d((1,256)) # don't change channel size, change width-height for fully layer

        # self.fully1 = nn.Linear(in_features=256, out_features=4096)
        # #self.fully2 = nn.Linear(in_features=4096, out_features=num_classes) # Use on BCE
        # self.fully2 = nn.Linear(in_features=4096, out_features=1) # for binary classification. Use on BCEWithLogitsLoss
        # self.fc = nn.Linear(256,1)
        
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=96,kernel_size=11,stride=4,padding=0)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.conv2 = nn.Conv2d(in_channels=96,out_channels=256,kernel_size=5,padding=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.conv3 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=2)
        self.conv4 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding=1)
        self.adaptivepool = nn.AdaptiveAvgPool2d((1,1))
        self.relu = nn.ReLU()
        self.fc = nn.Linear(256,1) #for binary classification

    def forward(self,x):
        # out = self.conv1(x)
        # out = self.relu(out)
        # out = self.maxpool1(out)

        # out = self.conv2(out)
        # out = self.relu(out)
        # out = self.maxpool2(out)

        # out = self.conv3(out)
        # out = self.conv4(out)
        # out = self.conv5(out)
        # out = self.relu(out)
        # out = self.maxpool3(out)

        # out = self.adaptivepool(out)

        # out = self.fully1(out)
        # print(out.shape)
        # out = self.fully2(out)#(out.view(out.shape[0],-1))
        # #out = self.fully2(out.view(out.shape[0],-1))
        # print(out.shape)
        out = self.conv1(x)
        out = self.relu(out)
        out = self.maxpool1(out)
        
        out = self.conv2(out)
        out = self.relu(out)
        out = self.maxpool2(out)
        
        out = self.conv3(out)
        out = self.relu(out) 
        
        out = self.conv4(out)
        out = self.relu(out)

        out = self.adaptivepool(out)
        out = self.fc(out.view(out.shape[0],-1))
        return out

#a = torch.rand(size=(2,3,128,128))
#model = AlexNet(2)
#out = model(a)
#print(out.shape)