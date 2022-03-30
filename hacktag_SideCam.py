import cv2 as cv
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import PIL.Image as image
from torchvision import models
import playsound
from threading import Thread

class ResNet34(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        # Use a pretrained model
        self.network = models.resnet34()
        # Replace last layer
        self.network.fc = nn.Linear(self.network.fc.in_features, num_classes)

    def forward(self,x):
        return self.network(x)

model = ResNet34(num_classes=10)

classes = ["Safe Driving","Possibly distracted","Possibly distracted","Possibly distracted","Possibly distracted","Possibly distracted","Possibly distracted","Possibly distracted","Possibly distracted","Possibly distracted"]
if torch.cuda.is_available():
    model.load_state_dict(torch.load("Hacktag_side_resnet34.pth",map_location=torch.device('cuda')))
else:
    model.load_state_dict(torch.load("Hacktag_side_resnet34.pth",map_location=torch.device('cpu')))
transforms = T.Compose([ T.Resize((64, 64)),
                               T.ColorJitter(brightness=.5, hue=.3),
                               T.ToTensor(),
                             ])
model.eval()
counter = 0
s=0

def sound(path):
    while(s):
        playsound.playsound(path) 
def classify(model,transforms,imge,classes):
    img = image.fromarray(imge)
    img = transforms(img).float()
    img = img[None, :]
    output = model(img)
    _,pred = torch.max(output.data,1)
    return pred.item()
    

cap = cv.VideoCapture(0)
count =0
while True:
    isTrue,frame = cap.read()
    if isTrue == 0:
        break 
    preds = classify(model,transforms,frame,classes)
    if preds== 0 : 
        counter=0
    else: counter = counter +1
    cv.putText(img=frame, text=classes[preds], org=(250, 350), fontFace=cv.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 255, 0),thickness=3)
    if counter>210:
        cv.putText(img=frame, text="CAUTION", org=(100, 100), fontFace=cv.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(255, 0, 0),thickness=3)
        s=1
        sound('beep.wav')
    

    cv.imshow('Video', frame)
    key = cv.waitKey(1)
    if key == 27:
      	break          

cap.release()
cv.destroyAllWindows()
