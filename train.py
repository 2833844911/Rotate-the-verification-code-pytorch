'''
author: cbb
email: 2833844911@qq.com
date: 2023-04-18
'''

import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import torch
import cv2
import random
import os
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision.models as models
from tqdm import tqdm
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# 将模型移动到GPU上（如果有可用的话）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 加载ResNet50模型
# resnet = models.resnet50(pretrained=True)
num_classes = 360


 # 导入图片大小
daorimg = (480, 640)

class CustomResNet50(nn.Module):
    def __init__(self, num_classes=num_classes):
        super(CustomResNet50, self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        self.resnet_layers = nn.Sequential(*list(resnet50.children())[:-1])
        self.flatten = nn.Flatten()
        # self.fc = nn.Linear(2048, num_classes)
        self.fc = nn.Linear(8192, num_classes)
        # self.sfm = nn.Softmax(1)

    def forward(self, x):
        # print(x.shape)
        x = self.resnet_layers(x)
        x = self.flatten(x)
        x = self.fc(x)
        # x = self.sfm(x)
        return x

resnet = CustomResNet50(num_classes=num_classes)
# resnet = torch.load('./modelyzm.pth')
resnet.to(device)

def get_jpg_paths(folder_path):
    jpg_paths = []
    for file in os.listdir(folder_path):
        jpg_paths.append(folder_path+'/'+file)
        
    return jpg_paths

folder_path = './imgs'
allfile = get_jpg_paths(folder_path)
random.shuffle(allfile)
huaf = int(len(allfile)*0.9)
tranfiles = allfile[:huaf]
textfiles = allfile[huaf:]
def add_gaussian_noise(image, mean=0, sigma=25):
    """
    添加高斯噪声到图像
    :param image: 输入图像
    :param mean: 噪声均值
    :param sigma: 噪声方差
    :return: 带有高斯噪声的图像
    """
    # 将图像转换为浮点数，以防止整数溢出
    image = np.array(image, dtype=np.float64)

    # 生成与图像尺寸相同的高斯噪声
    noise = np.random.normal(mean, sigma, image.shape)

    # 将噪声添加到图像
    noisy_image = image + noise

    # 确保结果在0到255范围内
    noisy_image = np.clip(noisy_image, 0, 255)

    # 将图像类型转换回原始类型
    noisy_image = noisy_image.astype(np.uint8)

    return noisy_image
class data_Dataset(Dataset):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data
    
    def __getitem__(self, index):
        tmfile = self.data[index]
        data = cv2.imread(tmfile)
        data = add_gaussian_noise(data)
       
        data =cv2.resize(data, daorimg, interpolation=cv2.INTER_AREA)
        angle =45
        # 计算旋转矩阵
        center = (data.shape[1] // 2, data.shape[0] // 2)
        angle = random.randint(0, num_classes-1)
        scale = 1
        M = cv2.getRotationMatrix2D(center, angle, scale)

        # 应用旋转矩阵
        rotated = cv2.warpAffine(data, M, (data.shape[1], data.shape[0]))
        
        # 中心圆圈裁剪
        mask = np.zeros((rotated.shape[0], rotated.shape[1]), dtype=np.uint8)
        x, y = np.meshgrid(np.arange(rotated.shape[1]), np.arange(rotated.shape[0]))
        dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        mask[dist <= rotated.shape[0] // 2] = 255
        cropped = cv2.bitwise_and(rotated, rotated, mask=mask)
        croppedcj = cropped[:, :]

        # 显示旋转后的图像
        # cv2.imshow('Rotated Image', croppedcj)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # 保存图片 
        cv2.imwrite('./test/{}_{}.jpg'.format(index, angle), croppedcj)

        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        img_tensor = transform(croppedcj)
        kk = img_tensor.shape[0]
        if kk != 3:

            img_tensor = torch.permute(img_tensor, (2, 0, 1))
        # print(img_tensor.shape)
        return img_tensor.to(device), torch.tensor( angle).to(device)

    def __len__(self):
        return len(self.data) -1


train_dataset = data_Dataset(tranfiles)

test_dataset = data_Dataset(textfiles)
batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class CustomClassificationLoss(nn.Module):
    def __init__(self, num_classes, alpha=18):
        super(CustomClassificationLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        # 计算交叉熵损失
        ce_loss = self.cross_entropy(logits, targets)

        # 计算距离损失
        probabilities = F.softmax(logits, dim=1)
        max_prob, _ = torch.max(probabilities, dim=1)
        daloss = torch.abs(_-targets)/360
        regularization = torch.mean(daloss)
        xiaoss = self.alpha * regularization
        # 组合损失
        loss = ce_loss + xiaoss
        return loss, xiaoss.item(), ce_loss.item()

# model = LeNet5()
criterion = CustomClassificationLoss(360)
optimizer = optim.Adam(resnet.parameters(), lr=0.0001)

max_xl = 1000

# 训练轮数
epochs = 100
for epoch in range(epochs):
    resnet.train()
    running_loss = 0
    xiao_loss = 0
    mmmm_loss = 0
    max_xlk = 0
    train_loader = tqdm(train_loader)
    for i, (inputs, labels) in enumerate(train_loader, 0):
        bashz = inputs.shape[0]
        if bashz != batch_size:
            continue
        optimizer.zero_grad()
        outputs = resnet(inputs)
        loss, xiao, mmmm = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        xiao_loss += xiao
        mmmm_loss += mmmm
        max_xlk =  running_loss/((i+1))
        train_loader.set_description(desc='Epoch %d loss: %.3f xiao_loss %.3f mmmm_loss: %.3f' % (epoch+1, running_loss/((i+1)), xiao_loss/((i+1)) , mmmm_loss/((i+1))))
    if max_xl > max_xlk:
        torch.save(resnet, './modelyzm.pth')
        max_xl = max_xlk



    lossall = 0
    alllen = 0
    
    with torch.no_grad():
        test_loader = tqdm(test_loader)
        resnet.eval()
        for data in test_loader:
            images, labels = data
            bashz = images.shape[0]
            if bashz != batch_size:
                continue
            alllen+=1
            outputs = resnet(images)
            loss, xiao, mmmm = criterion(outputs, labels)
            lossall += loss.item()

            test_loader.set_description(desc='test loss: {}'.format(lossall / alllen))

