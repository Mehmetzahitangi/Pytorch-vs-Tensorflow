import torch
from tqdm import tqdm

from custom_AlexNet import AlexNet
from dataset import CustomDataset

if __name__ == "__main__":
    batch_size = 5 # in each epoch number_data/batch_size will be processed
    num_epochs = 30 #120 
    num_classes = 2
    lr = 1e-4

    model = AlexNet(num_classes=num_classes)
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset = CustomDataset(data_path = "F:/yedek/00 AI-ML HER ŞEY/Pytorch vs Tensorflow/custom_alexnet/data/Train/")
    train_loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True) # DataLoader needs _getitem_ and _len_
    # DataLoader ==> If there are __getitem__ and __len__ then DataLoader can take the image

    optimizer = torch.optim.Adam(model.parameters(), lr =lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=2,factor=0.05,verbose=True)

    criterion = torch.nn.BCEWithLogitsLoss() #for binary classification. BinaryCrossEntropy ==> it needs 1 output not 2. In custom_alexnet fully connected output must be 1. 
    # criterion = torch.nn.BCELoss() # for binary classification with sigmoid func. it needs 2 output. In custom_alexnet fully connected output must be 2/num_classes. 

    for epoch in range(num_epochs):
        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        running_loss = []

        for batch_index, (img,target) in loop:
            img, target = img.to(device), target.to(device)

            output = model(img)
            target = target.unsqueeze(1).type(torch.float32) # for Binary Classification
            loss = criterion(output, target)

            optimizer.zero_grad() # Gradients on the optimizer is going to be 0
            loss.backward() # backpropagation
            optimizer.step() # update weights !?

            running_loss.append(loss.item())
            mean_loss = sum(running_loss) / len(running_loss)

            loop.set_description(f"Epoch: [{epoch+1}/{num_epochs}]")
            loop.set_postfix(batch_loss = loss.item(), mean_loss=mean_loss, lr = optimizer.param_groups[0]["lr"])

        torch.save(model.state_dict(), "./binary_model.pth")
        scheduler.step(mean_loss) # update learning rate