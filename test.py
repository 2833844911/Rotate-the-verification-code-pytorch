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
# 加载ResNet50模型
# resnet = models.resnet50(pretrained=True)
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
        # print(x.shape)
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
    img = crop_max_square(img)
    rotated =cv2.resize(img, (480, 480), interpolation=cv2.INTER_AREA)
    center = (rotated.shape[1] // 2, rotated.shape[0] // 2)
    mask = np.zeros((rotated.shape[0], rotated.shape[1]), dtype=np.uint8)
    x, y = np.meshgrid(np.arange(rotated.shape[1]), np.arange(rotated.shape[0]))
    dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    mask[dist <= rotated.shape[0] // 2] = 255
    img = cv2.bitwise_and(rotated, rotated, mask=mask)
    # cv2.imshow('Rotated Image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
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



resnet = torch.load('./modelyzm.pth')
resnet.to(device)

# 需要识别的图片
pathfile = './test/49_320.jpg'
imgTensor = getImgae(pathfile)
# print(imgTensor.shape)
outInfo = resnet(imgTensor)
k,o = torch.max(outInfo, dim=1)

# 展示图片
angle =360-o[0].item()
print("旋转角度",angle)
# 计算旋转矩阵
data = cv2.imread(pathfile)
center = (data.shape[1] // 2, data.shape[0] // 2)
scale = 1
M = cv2.getRotationMatrix2D(center, angle, scale)

# 应用旋转矩阵
rotated = cv2.warpAffine(data, M, (data.shape[1], data.shape[0]))
cv2.imshow('original', data)

cv2.imshow('Rotated Image', rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
