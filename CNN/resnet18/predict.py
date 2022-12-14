import os
from cv2 import transform
import torch
import torchvision.transforms
import numpy as np
import cv2
from PIL import Image

from resnet_model import ResNet18


def predict_multiclass(model,img):
    img = img.unsqueeze(0)#.cuda()
    logit = model(img)
    print("Logit: ",logit)
    probability = torch.softmax(logit,dim=1)[0].cpu().detach().numpy()
    print("Probability: ",probability)
    prediction = logit.argmax().cpu().numpy()
    print("Prediction: ",prediction)
    return prediction,probability,logit

def predict_binary(model,img):

    img = img.unsqueeze(0)#.cuda()
    logit = model(img)
    print("Raw Logit: ",logit)
    probability = torch.sigmoid(logit).cpu().detach().numpy()
    print("Probability: ",probability[0][0])

    if probability[0][0] < 0.5:
        prediction = 0

    elif probability[0][0] >= 0.5:
        prediction = 1

    return prediction,probability,logit

if __name__ == "__main__":
    num_classes = 3
    input_size = (256,256)
    test_images_path = "./resnet18/Rock-Paper-Scissors/test/scissors/"
    model_path = "resnet18/checkpoints/model_rock_paper.pth"
    #test_data_path = "./custom_alexnet/data/Test/"

    transform = torchvision.transforms.Compose([torchvision.transforms.Resize(size = (input_size)),
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResNet18(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    test_images = os.listdir(test_images_path)

    for index, image_name in enumerate(test_images):
        img = Image.open(test_images_path + image_name).convert('RGB')
        vis_image = np.array(img)
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)

        img = transform(img)
        img.to(device)

        with torch.no_grad(): # stop update gradients
            #prediction,probability,logit = predict_binary(model,img) 
            prediction,probability,logit = predict_multiclass(model,img) 
            cv2.putText(vis_image,str(prediction),(10,10),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),1)

        cv2.imshow("img", vis_image)
        key = cv2.waitKey(0)

        if key == ord("q"):
            break
        elif key == ord("d"):
            continue 
