import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from resnet_model import ResNet18

if __name__ == "__main__":
    input_size = (256,256) #height,width
    #batch_size = 3
    num_epochs = 80
    num_classes = 3
    lr = 1e-4
    train_data = "F:/yedek/00 AI-ML HER ŞEY/Pytorch vs Tensorflow/CNN/resnet18/Rock-Paper-Scissors/train"

    model = ResNet18(num_classes= num_classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    data_transforms = transforms.Compose([transforms.Resize(size = (input_size)),
                                          transforms.ToTensor(),
                                          transforms. Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    trainset = torchvision.datasets.ImageFolder(train_data, transform= data_transforms)
    train_loader = torch.utils.data.DataLoader(dataset = trainset, batch_size = 2, shuffle = True)

    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.05, verbose=True)
    criterion = torch.nn.CrossEntropyLoss()
    #criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        loop = tqdm(enumerate(train_loader), total =len(train_loader))
        running_loss = []

        for batch_index, (data,target) in loop:
            img, target = data.to(device), target.to(device)

            output = model(img)
            # target = target.unsqueeze(1).type(torch.float32) # for Binary Classification
            loss = criterion(output, target)

            #print("Target shape: ",target.shape)
            #print("Output shape : ",output.shape)

            optimizer.zero_grad() # Gradients on the optimizer is going to be 0
            loss.backward() # backpropagation
            optimizer.step() # update weights !?

            running_loss.append(loss.item())
            mean_loss = sum(running_loss) / len(running_loss)

            loop.set_description(f"Epoch: [{epoch+1}/{num_epochs}]")
            loop.set_postfix(batch_loss = loss.item(), mean_loss=mean_loss, lr = optimizer.param_groups[0]["lr"])

        torch.save(model.state_dict(), "F:/yedek/00 AI-ML HER ŞEY/Pytorch vs Tensorflow/CNN/resnet18/checkpoints/model_rock_paper.pth")
        scheduler.step(mean_loss) # update learning rate