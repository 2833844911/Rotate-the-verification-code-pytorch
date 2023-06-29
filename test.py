import torchvision.transforms as transforms
import torch
import cv2
import os
import torch.nn as nn
import numpy as np
import torchvision.models as models
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# 将模型移动到GPU上（如果有可用的话）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 360

class CustomResNet50(nn.Module):
    def __init__(self, num_classes=num_classes):
        super(CustomResNet50, self).__init__()
        resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet_layers = nn.Sequential(*list(resnet50.children())[:-1])
        self.flatten = nn.Flatten()
        # self.fc = nn.Linear(2048, num_classes)
        self.fc = nn.Linear(8192, num_classes)


    def forward(self, x):
        x = self.resnet_layers(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

def crop_max_square(img):
    size = min(img.shape[:2])
    x = (img.shape[1] - size) // 2
    y = (img.shape[0] - size) // 2
    return img[y:y+size, x:x+size]
def getImgae(path):
    print(path)
    img = cv2.imread(path)

    rotated =cv2.resize(img, (480, 480), interpolation=cv2.INTER_AREA)

    img = rotated

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    img_tensor = transform(img)
    kk = img_tensor.shape[0]
    if kk != 3:
        img_tensor = torch.permute(img_tensor, (2, 0, 1))
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.to(device)
    return img_tensor

def getDu(data):
    k = data.tolist()[0]
    for i in range(len(k)):
        k[i] = [k[i], i]
    k.sort()
    k = k[::-1]

    print()
    return k[0][1]


resnet = torch.load('./modelyzm.pth')
resnet.to(device)

# 需要识别的图片
pathfile = './test/5_110.jpg'
# pathfile = './test/104_257.jpg'
# pathfile = './test/105_101.jpg'
# pathfile = './test/115_101.jpg'
# pathfile = './test/294_181.jpg'
# pathfile = './test/321_219.jpg'
imgTensor = getImgae(pathfile)
# print(imgTensor.shape)
outInfo = resnet(imgTensor)
o = getDu(outInfo)


# 展示图片
angle = o
print("旋转角度",angle)
# 计算旋转矩阵
data = cv2.imread(pathfile)
center = (data.shape[1] // 2, data.shape[0] // 2)
scale = 1
M = cv2.getRotationMatrix2D(center,  360 - angle, scale)

# 应用旋转矩阵
rotated = cv2.warpAffine(data, M, (data.shape[1], data.shape[0]))
cv2.imshow('original', data)

cv2.imshow('Rotated Image', rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
